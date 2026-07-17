package videojobs

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/openai"
)

// mockTranscoder is a Transcoder that records calls and returns canned output.
type mockTranscoder struct {
	available bool
	encodeErr error
	mu        sync.Mutex
	calls     []struct {
		frames int
		fps    int
	}
	output      []byte
	decodeFrames [][]byte
	decodeErr    error
	concatErr    error
	concatOutput []byte
}

func (m *mockTranscoder) EncodeMP4(ctx context.Context, framePNGs [][]byte, fps int) ([]byte, error) {
	m.mu.Lock()
	m.calls = append(m.calls, struct {
		frames int
		fps    int
	}{len(framePNGs), fps})
	m.mu.Unlock()
	if m.encodeErr != nil {
		return nil, m.encodeErr
	}
	if m.output != nil {
		return m.output, nil
	}
	// Return a stub MP4 (ftyp box header) so the result is non-empty.
	return []byte{0, 0, 0, 0x18, 'f', 't', 'y', 'p'}, nil
}

// DecodeFrames returns the canned decodeFrames (or a single stub frame if nil)
// so edit/extend worker tests can exercise the full path without a real MP4.
func (m *mockTranscoder) DecodeFrames(ctx context.Context, mp4 []byte, maxFrames int) ([][]byte, int, error) {
	if m.decodeErr != nil {
		return nil, 0, m.decodeErr
	}
	if m.decodeFrames != nil {
		return m.decodeFrames, 16, nil
	}
	// Default: one small stub PNG frame.
	return [][]byte{[]byte("stub-frame")}, 16, nil
}

// DecodeLastFrame returns the last canned frame (or a stub).
func (m *mockTranscoder) DecodeLastFrame(ctx context.Context, mp4 []byte) ([]byte, error) {
	if m.decodeErr != nil {
		return nil, m.decodeErr
	}
	if m.decodeFrames != nil && len(m.decodeFrames) > 0 {
		return m.decodeFrames[len(m.decodeFrames)-1], nil
	}
	return []byte("stub-last-frame"), nil
}

// ConcatMP4 returns the canned concat output, or a stub for tests.
func (m *mockTranscoder) ConcatMP4(ctx context.Context, first, second []byte, fps int) ([]byte, error) {
	if m.concatErr != nil {
		return nil, m.concatErr
	}
	if m.concatOutput != nil {
		return m.concatOutput, nil
	}
	return []byte{0, 0, 0, 0x18, 'f', 't', 'y', 'p'}, nil
}

func (m *mockTranscoder) Available() bool { return m.available }

func (m *mockTranscoder) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.calls)
}

// genFuncBuilder helps construct a GenerateFunc that emits a scripted sequence
// of frames and progress updates.
type genFuncBuilder struct {
	frames   [][]byte
	steps    []struct{ step, total int }
	err      error
	delay    time.Duration
	started  atomic.Bool
	cancelCh chan struct{}
}

func newGenFunc() *genFuncBuilder {
	return &genFuncBuilder{cancelCh: make(chan struct{})}
}

func (g *genFuncBuilder) withFrames(n int) *genFuncBuilder {
	for i := 0; i < n; i++ {
		g.frames = append(g.frames, []byte{byte(i)})
	}
	return g
}

func (g *genFuncBuilder) withStep(step, total int) *genFuncBuilder {
	g.steps = append(g.steps, struct{ step, total int }{step, total})
	return g
}

func (g *genFuncBuilder) withError(err error) *genFuncBuilder {
	g.err = err
	return g
}

func (g *genFuncBuilder) withDelay(d time.Duration) *genFuncBuilder {
	g.delay = d
	return g
}

func (g *genFuncBuilder) build() GenerateFunc {
	return func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		g.started.Store(true)
		for _, s := range g.steps {
			fn(nil, s.step, s.total)
			if g.delay > 0 {
				select {
				case <-time.After(g.delay):
				case <-ctx.Done():
					return ctx.Err()
				case <-g.cancelCh:
					return errors.New("cancelled by test")
				}
			}
		}
		for _, f := range g.frames {
			fn(f, 0, 0)
		}
		return g.err
	}
}

func waitForStatus(t *testing.T, j *Job, want string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if j.Status() == want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for status %q, got %q", want, j.Status())
}

func TestCreateJobCompletes(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc().withFrames(3).withStep(2, 10).withStep(10, 10)
	j, err := store.Create(CreateParams{
		Model:    "wan2.1-t2v",
		Prompt:   "a cat",
		Seconds:  "4",
		Size:     "720x1280",
		Generate: g.build(),
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	if j.Status() != openai.VideoStatusQueued && j.Status() != openai.VideoStatusInProgress && j.Status() != openai.VideoStatusCompleted {
		t.Fatalf("initial status unexpected: %q", j.Status())
	}

	waitForStatus(t, j, openai.VideoStatusCompleted, 5*time.Second)

	v := j.ToVideo()
	if v.Status != openai.VideoStatusCompleted {
		t.Errorf("status = %q, want completed", v.Status)
	}
	if v.Progress != 100 {
		t.Errorf("progress = %d, want 100", v.Progress)
	}
	if v.ID == "" || v.Object != openai.VideoObject {
		t.Errorf("id/object not set: id=%q object=%q", v.ID, v.Object)
	}
	if v.Model != "wan2.1-t2v" || v.Prompt != "a cat" || v.Seconds != "4" || v.Size != "720x1280" {
		t.Errorf("params not echoed: %+v", v)
	}
	if v.CompletedAt == 0 || v.ExpiresAt == 0 {
		t.Errorf("completed_at/expires_at not set: completed=%d expires=%d", v.CompletedAt, v.ExpiresAt)
	}

	content, ct := j.Content()
	if len(content) == 0 {
		t.Fatal("expected non-empty MP4 content")
	}
	if ct != "video/mp4" {
		t.Errorf("content type = %q, want video/mp4", ct)
	}

	if tc.callCount() != 1 {
		t.Errorf("transcoder called %d times, want 1", tc.callCount())
	}
}

func TestCreateJobFailsOnGenerationError(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc().withFrames(1).withError(errors.New("model OOM"))
	j, err := store.Create(CreateParams{
		Model:    "wan2.1-t2v",
		Prompt:   "a cat",
		Seconds:  "4",
		Size:     "720x1280",
		Generate: g.build(),
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	waitForStatus(t, j, openai.VideoStatusFailed, 5*time.Second)

	v := j.ToVideo()
	if v.Status != openai.VideoStatusFailed {
		t.Fatalf("status = %q, want failed", v.Status)
	}
	if v.Error == nil {
		t.Fatal("expected error payload")
	}
	if v.Error.Code != "generation_failed" {
		t.Errorf("error code = %q, want generation_failed", v.Error.Code)
	}
	if v.Error.Message == "" {
		t.Error("error message empty")
	}

	if content, _ := j.Content(); content != nil {
		t.Error("expected no content on failed job")
	}
}

func TestCreateJobFailsWhenNoFrames(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc() // emits nothing
	j, _ := store.Create(CreateParams{
		Model:    "wan2.1-t2v",
		Prompt:   "a cat",
		Seconds:  "4",
		Size:     "720x1280",
		Generate: g.build(),
	})

	waitForStatus(t, j, openai.VideoStatusFailed, 5*time.Second)

	v := j.ToVideo()
	if v.Error == nil || v.Error.Code != "no_frames" {
		t.Fatalf("expected no_frames error, got %+v", v.Error)
	}
}

func TestCreateJobFailsWhenTranscoderUnavailable(t *testing.T) {
	tc := &mockTranscoder{available: false}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc().withFrames(2)
	j, _ := store.Create(CreateParams{
		Model:    "wan2.1-t2v",
		Prompt:   "a cat",
		Seconds:  "4",
		Size:     "720x1280",
		Generate: g.build(),
	})

	waitForStatus(t, j, openai.VideoStatusFailed, 5*time.Second)

	v := j.ToVideo()
	if v.Error == nil || v.Error.Code != "ffmpeg_required" {
		t.Fatalf("expected ffmpeg_required error, got %+v", v.Error)
	}
}

func TestCreateJobFailsOnEncodeError(t *testing.T) {
	tc := &mockTranscoder{available: true, encodeErr: errors.New("encoder broken")}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc().withFrames(2)
	j, _ := store.Create(CreateParams{
		Model:    "wan2.1-t2v",
		Prompt:   "a cat",
		Seconds:  "4",
		Size:     "720x1280",
		Generate: g.build(),
	})

	waitForStatus(t, j, openai.VideoStatusFailed, 5*time.Second)

	v := j.ToVideo()
	if v.Error == nil || v.Error.Code != "encoding_failed" {
		t.Fatalf("expected encoding_failed error, got %+v", v.Error)
	}
}

func TestDeleteCancelsRunningJob(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc().withStep(1, 10).withDelay(10 * time.Second)
	j, _ := store.Create(CreateParams{
		Model:    "wan2.1-t2v",
		Prompt:   "a cat",
		Seconds:  "4",
		Size:     "720x1280",
		Generate: g.build(),
	})

	// Wait for the job to be in progress (past the semaphore, into Generate).
	waitForStatus(t, j, openai.VideoStatusInProgress, 5*time.Second)

	deleted, err := store.Delete(j.ID())
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if !deleted {
		t.Fatal("expected deleted=true")
	}

	// Job should no longer be retrievable.
	_, err = store.Get(j.ID())
	if !errors.Is(err, ErrJobNotFound) {
		t.Errorf("expected ErrJobNotFound after delete, got %v", err)
	}
}

func TestDeleteNonexistent(t *testing.T) {
	store := NewJobStore(&mockTranscoder{available: true})
	defer store.Close()

	_, err := store.Delete("vid_nonexistent")
	if !errors.Is(err, ErrJobNotFound) {
		t.Errorf("expected ErrJobNotFound, got %v", err)
	}
}

func TestGetNonexistent(t *testing.T) {
	store := NewJobStore(&mockTranscoder{available: true})
	defer store.Close()

	_, err := store.Get("vid_nonexistent")
	if !errors.Is(err, ErrJobNotFound) {
		t.Errorf("expected ErrJobNotFound, got %v", err)
	}
}

func TestListOrderAndPagination(t *testing.T) {
	store := NewJobStore(&mockTranscoder{available: true})
	defer store.Close()

	// Create 5 jobs that complete instantly.
	var jobs []*Job
	for i := 0; i < 5; i++ {
		g := newGenFunc().withFrames(1)
		j, _ := store.Create(CreateParams{
			Model: "wan2.1-t2v", Prompt: "p", Seconds: "4", Size: "720x1280",
			Generate: g.build(),
		})
		jobs = append(jobs, j)
	}
	// Wait for all to complete so listing is stable.
	for _, j := range jobs {
		waitForStatus(t, j, openai.VideoStatusCompleted, 5*time.Second)
	}

	// Default (desc) order, limit 2.
	got, hasMore := store.List("", 2, "desc")
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if !hasMore {
		t.Error("expected has_more=true")
	}
	// Desc: most recent first.
	if got[0].ID() != jobs[4].ID() {
		t.Errorf("first id = %q, want %q (desc order)", got[0].ID(), jobs[4].ID())
	}

	// Asc order, limit 3.
	got, hasMore = store.List("", 3, "asc")
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	if !hasMore {
		t.Error("expected has_more=true")
	}
	if got[0].ID() != jobs[0].ID() {
		t.Errorf("first id = %q, want %q (asc order)", got[0].ID(), jobs[0].ID())
	}

	// Cursor: after the first asc item.
	got, hasMore = store.List(jobs[0].ID(), 10, "asc")
	if len(got) != 4 {
		t.Fatalf("len = %d, want 4", len(got))
	}
	if hasMore {
		t.Error("expected has_more=false")
	}
	if got[0].ID() != jobs[1].ID() {
		t.Errorf("first id = %q, want %q", got[0].ID(), jobs[1].ID())
	}
}

func TestListEmptyStore(t *testing.T) {
	store := NewJobStore(&mockTranscoder{available: true})
	defer store.Close()

	got, hasMore := store.List("", 10, "desc")
	if len(got) != 0 {
		t.Fatalf("len = %d, want 0", len(got))
	}
	if hasMore {
		t.Error("expected has_more=false on empty store")
	}
}

// TestListDeletedCursorReturnsAll verifies that when the `after` cursor job
// has been deleted/evicted between paginated calls, List does not return an
// empty page (the desc-bug regression) but degrades to returning all entries
// (so pagination restarts rather than losing data).
func TestListDeletedCursorReturnsAll(t *testing.T) {
	store := NewJobStore(&mockTranscoder{available: true})
	defer store.Close()

	var jobs []*Job
	for i := 0; i < 5; i++ {
		g := newGenFunc().withFrames(1)
		j, _ := store.Create(CreateParams{
			Model: "wan2.1-t2v", Prompt: "p", Seconds: "4", Size: "720x1280",
			Generate: g.build(),
		})
		jobs = append(jobs, j)
	}
	for _, j := range jobs {
		waitForStatus(t, j, openai.VideoStatusCompleted, 5*time.Second)
	}

	// Delete the job that would be the cursor (jobs[2]).
	if _, err := store.Delete(jobs[2].ID()); err != nil {
		t.Fatalf("Delete cursor job: %v", err)
	}

	// Paginate with the deleted cursor: desc should NOT return empty.
	got, hasMore := store.List(jobs[2].ID(), 10, "desc")
	if len(got) == 0 {
		t.Fatal("desc returned empty page for deleted cursor; expected all remaining entries")
	}
	// The deleted job itself must not appear.
	for _, j := range got {
		if j.ID() == jobs[2].ID() {
			t.Error("deleted cursor job appeared in results")
		}
	}
	_ = hasMore

	// Same for asc.
	got, _ = store.List(jobs[2].ID(), 10, "asc")
	if len(got) == 0 {
		t.Fatal("asc returned empty page for deleted cursor; expected all remaining entries")
	}
}

func TestProgressUpdates(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	// Emit progress 5/10 (50%), then 10/10. Worker should cap at 99 until
	// completed.
	g := newGenFunc().withStep(5, 10).withStep(10, 10).withFrames(1)
	j, _ := store.Create(CreateParams{
		Model: "wan2.1-t2v", Prompt: "p", Seconds: "4", Size: "720x1280",
		Generate: g.build(),
	})

	waitForStatus(t, j, openai.VideoStatusCompleted, 5*time.Second)
	v := j.ToVideo()
	if v.Progress != 100 {
		t.Errorf("final progress = %d, want 100", v.Progress)
	}
}

func TestRemixedFromID(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	g := newGenFunc().withFrames(1)
	j, _ := store.Create(CreateParams{
		Model:         "wan2.1-t2v",
		Prompt:        "p",
		Seconds:       "4",
		Size:          "720x1280",
		RemixedFromID: "vid_abc123",
		Generate:      g.build(),
	})
	waitForStatus(t, j, openai.VideoStatusCompleted, 5*time.Second)

	v := j.ToVideo()
	if v.RemixedFromVideoID != "vid_abc123" {
		t.Errorf("remixed_from_video_id = %q, want vid_abc123", v.RemixedFromVideoID)
	}
}

func TestCreateRequiresGenerateFunc(t *testing.T) {
	store := NewJobStore(&mockTranscoder{available: true})
	defer store.Close()

	_, err := store.Create(CreateParams{
		Model: "wan2.1-t2v", Prompt: "p", Seconds: "4", Size: "720x1280",
	})
	if err == nil {
		t.Fatal("expected error when Generate is nil")
	}
}

// TestEditJobUsesFirstFrameAsInitImage verifies that an edit job (Extend=false)
// with a SourceVideoID resolves the source's MP4, decodes its frames, and
// passes the FIRST frame as InitImage to the Generate func.
func TestEditJobUsesFirstFrameAsInitImage(t *testing.T) {
	tc := &mockTranscoder{
		available:    true,
		decodeFrames: [][]byte{[]byte("first-frame"), []byte("second-frame"), []byte("third-frame")},
	}
	store := NewJobStore(tc)
	defer store.Close()

	// First, create a completed source job to reference.
	var capturedInitImage []byte
	sourceGen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		fn([]byte("source-frame"), 0, 0)
		return nil
	}
	srcJob, err := store.Create(CreateParams{
		Model: "wan2.1-t2v", Prompt: "source", Seconds: "4", Size: "720x1280",
		Generate: sourceGen,
	})
	if err != nil {
		t.Fatalf("create source: %v", err)
	}
	waitForStatus(t, srcJob, openai.VideoStatusCompleted, 5*time.Second)

	// Now create an edit job referencing the source.
	editGen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		capturedInitImage = params.InitImage
		fn([]byte("edited-frame"), 0, 0)
		return nil
	}
	editJob, err := store.Create(CreateParams{
		Model:         "wan2.1-t2v",
		Prompt:        "edited",
		Seconds:       "4",
		Size:          "720x1280",
		SourceVideoID: srcJob.ID(),
		RemixedFromID: srcJob.ID(),
		Extend:        false,
		Generate:      editGen,
	})
	if err != nil {
		t.Fatalf("create edit: %v", err)
	}
	waitForStatus(t, editJob, openai.VideoStatusCompleted, 5*time.Second)

	if string(capturedInitImage) != "first-frame" {
		t.Errorf("edit init image = %q, want %q (first frame)", string(capturedInitImage), "first-frame")
	}

	v := editJob.ToVideo()
	if v.RemixedFromVideoID != srcJob.ID() {
		t.Errorf("remixed_from_video_id = %q, want %q", v.RemixedFromVideoID, srcJob.ID())
	}
}

// TestExtendJobUsesLastFrameAsInitImage verifies that an extension job
// (Extend=true) passes the LAST frame as InitImage (via DecodeLastFrame) and
// stitches the source + generated via ConcatMP4, with seconds = source + requested.
func TestExtendJobUsesLastFrameAsInitImage(t *testing.T) {
	tc := &mockTranscoder{
		available:    true,
		decodeFrames: [][]byte{[]byte("f1"), []byte("f2"), []byte("f3-last")},
		concatOutput: []byte("stitched-mp4"),
	}
	store := NewJobStore(tc)
	defer store.Close()

	// Create a completed source job with seconds="8".
	sourceGen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		fn([]byte("source-frame"), 0, 0)
		return nil
	}
	srcJob, err := store.Create(CreateParams{
		Model: "wan2.1-t2v", Prompt: "source", Seconds: "8", Size: "720x1280",
		Generate: sourceGen,
	})
	if err != nil {
		t.Fatalf("create source: %v", err)
	}
	waitForStatus(t, srcJob, openai.VideoStatusCompleted, 5*time.Second)

	var capturedInitImage []byte
	extGen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		capturedInitImage = params.InitImage
		fn([]byte("ext-frame"), 0, 0)
		return nil
	}
	extJob, err := store.Create(CreateParams{
		Model:         "wan2.1-t2v",
		Prompt:        "extend",
		Seconds:       "4",
		Size:          "720x1280",
		SourceVideoID: srcJob.ID(),
		RemixedFromID: srcJob.ID(),
		Extend:        true,
		SourceSeconds: "8",
		Generate:      extGen,
	})
	if err != nil {
		t.Fatalf("create extension: %v", err)
	}
	waitForStatus(t, extJob, openai.VideoStatusCompleted, 5*time.Second)

	// Init image should be the LAST frame via DecodeLastFrame.
	if string(capturedInitImage) != "f3-last" {
		t.Errorf("extend init image = %q, want %q (last frame via DecodeLastFrame)", string(capturedInitImage), "f3-last")
	}

	v := extJob.ToVideo()
	// Stitched seconds = 8 (source) + 4 (requested) = 12.
	if v.Seconds != "12" {
		t.Errorf("stitched seconds = %q, want %q", v.Seconds, "12")
	}
	// Content should be the concat output, not the generated segment alone.
	content, _ := extJob.Content()
	if string(content) != "stitched-mp4" {
		t.Errorf("content = %q, want %q (stitched)", string(content), "stitched-mp4")
	}
}

// TestExtendJobWithFileUpload verifies the SourceVideo (uploaded file) path
// also decodes the last frame (via DecodeLastFrame) and stitches.
func TestExtendJobWithFileUpload(t *testing.T) {
	tc := &mockTranscoder{
		available:    true,
		concatOutput: []byte("stitched"),
	}
	store := NewJobStore(tc)
	defer store.Close()

	var capturedInitImage []byte
	extGen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		capturedInitImage = params.InitImage
		fn([]byte("gen"), 0, 0)
		return nil
	}
	extJob, err := store.Create(CreateParams{
		Model:         "wan2.1-t2v",
		Prompt:        "extend",
		Seconds:       "4",
		Size:          "720x1280",
		SourceVideo:   []byte("uploaded-mp4-bytes"),
		Extend:        true,
		SourceSeconds: "0", // unknown for file uploads
		Generate:      extGen,
	})
	if err != nil {
		t.Fatalf("create extension: %v", err)
	}
	waitForStatus(t, extJob, openai.VideoStatusCompleted, 5*time.Second)

	// DecodeLastFrame returns the stub "stub-last-frame" when decodeFrames is nil.
	if string(capturedInitImage) != "stub-last-frame" {
		t.Errorf("extend init image = %q, want %q", string(capturedInitImage), "stub-last-frame")
	}

	v := extJob.ToVideo()
	// SourceSeconds unknown (0) → falls back to requested seconds.
	if v.Seconds != "4" {
		t.Errorf("seconds = %q, want %q (fallback to requested)", v.Seconds, "4")
	}
}

// TestEditJobFailsOnUnknownSourceID verifies that referencing a nonexistent
// source job id fails the new job with source_video_unavailable.
func TestEditJobFailsOnUnknownSourceID(t *testing.T) {
	tc := &mockTranscoder{available: true}
	store := NewJobStore(tc)
	defer store.Close()

	editGen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		fn([]byte("frame"), 0, 0)
		return nil
	}
	editJob, _ := store.Create(CreateParams{
		Model: "wan2.1-t2v", Prompt: "p", Seconds: "4", Size: "720x1280",
		SourceVideoID: "vid_nonexistent", Extend: false, Generate: editGen,
	})
	waitForStatus(t, editJob, openai.VideoStatusFailed, 5*time.Second)

	v := editJob.ToVideo()
	if v.Error == nil || v.Error.Code != "source_video_unavailable" {
		t.Fatalf("expected source_video_unavailable error, got %+v", v.Error)
	}
}

// TestStitchSeconds verifies the seconds-stitching helper.
func TestStitchSeconds(t *testing.T) {
	cases := []struct {
		source, requested, want string
	}{
		{"8", "4", "12"},
		{"4", "4", "8"},
		{"12", "8", "20"},
		{"", "4", "4"},     // unknown source → requested
		{"abc", "4", "4"},  // invalid source → requested
		{"8", "", ""},      // invalid requested → empty
	}
	for _, c := range cases {
		got := stitchSeconds(c.source, c.requested)
		if got != c.want {
			t.Errorf("stitchSeconds(%q, %q) = %q, want %q", c.source, c.requested, got, c.want)
		}
	}
}
