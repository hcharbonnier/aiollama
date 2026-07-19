// Package videojobs implements an in-memory job store and worker for the
// OpenAI Videos API (/v1/videos). The OpenAI Videos API is asynchronous: a
// POST /v1/videos request creates a job (status "queued"), the client polls
// GET /v1/videos/{id} until status is "completed" or "failed", then fetches
// the binary MP4 via GET /v1/videos/{id}/content.
//
// This package bridges that async model with the synchronous, streaming
// diffgen runner (x/diffgen). A worker goroutine drives the runner's
// Completion call, collects the streamed frames, transcodes them to MP4 via
// ffmpeg, and stores the result on the job for later download.
//
// The job store is in-memory and process-local: jobs are lost on restart.
// The JobStore interface allows a persistent (disk/SQLite) implementation to
// drop in later without touching the HTTP handlers. An expires_at TTL bounds
// retention per the spec.
package videojobs

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/ollama/ollama/openai"
)

// MaxConcurrentJobs bounds how many video generation jobs may run
// simultaneously. Video generation is resource-heavy (the diffgen runner
// serializes per-model via diffGenMu); excess POST /v1/videos requests are
// accepted (status "queued") and wait for a worker slot. Default 1; use
// NewJobStoreWithConcurrency to override (e.g. from an env var).
const MaxConcurrentJobs = 1

// JobTTL is how long a completed/failed job (and its content) is retained
// after completion before being eligible for expiry-driven eviction. The spec
// exposes expires_at; clients should not rely on content being available
// beyond this window.
const JobTTL = 30 * time.Minute

// MaxJobAge bounds how long a non-terminal job may run before it is
// force-failed. This prevents a hung runner from leaking a job (and its
// concurrency slot) indefinitely.
const MaxJobAge = 2 * time.Hour

// MaxTotalContentBytes bounds the aggregate retained MP4 content across all
// completed jobs. When exceeded, the oldest completed jobs are evicted early
// (before their TTL) to free memory.
const MaxTotalContentBytes int64 = 2 * 1024 * 1024 * 1024 // 2 GiB

// Job is an in-flight or completed video generation job. It is the internal
// representation; the HTTP-facing shape is openai.Video (see ToVideo).
type Job struct {
	id            string
	model         string
	prompt        string
	seconds       string
	size          string
	remixedFromID string
	createdAt     int64
	seqno         int64 // monotonic creation order, for stable pagination
	completedAt   int64
	expiresAt     int64

	mu       sync.RWMutex
	status   string
	progress int
	err      *openai.VideoError

	content     []byte
	contentType string

	// Cached thumbnail/spritesheet variants (PNG), computed lazily on first
	// GET /content?variant=... The source MP4 is immutable once completed,
	// so variants never need recomputation.
	thumbnail   []byte
	spritesheet []byte

	cancel     context.CancelFunc
	cancelOnce sync.Once
	done       chan struct{}
}

// cancelCtx cancels the job's worker context exactly once. Safe to call
// from Delete, Close, or the worker's own defer.
func (j *Job) cancelCtx() {
	j.cancelOnce.Do(func() {
		if j.cancel != nil {
			j.cancel()
		}
	})
}

// CreateParams holds the validated parameters for a new video job. The
// caller (HTTP handler) is responsible for parsing multipart fields and
// resolving input_reference into the optional InitImage before calling Create.
type CreateParams struct {
	Model         string
	Prompt        string
	Seconds       string
	Size          string
	RemixedFromID string
	// InitImage is the optional decoded reference image (from
	// input_reference) for image-to-video generation. May be nil for
	// text-to-video.
	InitImage []byte
	// Generate is the function the worker calls to drive the runner. It is
	// injected so the package does not depend on the scheduler directly,
	// enabling unit tests with a mock. See Worker.GenerateFunc.
	Generate GenerateFunc

	// --- Edit / Extension parameters (Phase 2) ---

	// SourceVideoID references a previously-completed job whose MP4 is the
	// input to an edit or extension. The worker decodes its frames to derive
	// the I2V init image (edit: first frame) or the continuation point
	// (extension: last frame). Mutually exclusive with SourceVideo (an
	// uploaded file). Empty for plain create.
	SourceVideoID string

	// SourceVideo is an uploaded source MP4 (from a multipart file part) for
	// edit/extension. Mutually exclusive with SourceVideoID. The worker
	// does NOT store this on the job (the resulting video is what's kept);
	// it decodes frames from it for the init image.
	SourceVideo []byte

	// Extend indicates this is an extension (not an edit). When true:
	//   - the worker uses the LAST frame of the source as the init image.
	//   - the worker concatenates the source MP4 + the generated extension
	//     via ConcatMP4, and the resulting Video.seconds is the stitched
	//     total (source seconds + requested seconds).
	// When false (edit): the worker uses the FIRST frame as the init image
	// and discards the source after generation (the result is the new
	// generation only, per the Sora edits semantics: re-render from a
	// reference frame with a new prompt).
	Extend bool

	// SourceSeconds is the duration of the source video, used to compute the
	// stitched total seconds for extensions. The caller fills this from the
	// referenced job's Seconds field (for SourceVideoID) or by parsing the
	// uploaded file (for SourceVideo — left 0/unknown in v1). For edits, it
	// is ignored.
	SourceSeconds string
}

// GenerateFunc drives the diffgen runner for a single video generation,
// invoking fn for each streamed response. It is the contract the worker uses
// to run generation; the real implementation (in server/videoapi.go) calls
// scheduleRunner + runner.Completion. The callback receives the raw RGB frame
// PNG bytes (base64-decoded) for each streamed frame and the current
// step/total progress. The final frame is also delivered via the callback
// before done. Returns an error if generation failed.
//
// framePNGs is populated with every frame's decoded PNG bytes in order; the
// worker uses this to transcode to MP4. Progress updates (step/total) are
// delivered via the progress callback.
type GenerateFunc func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error

// ToVideo renders the job's public OpenAI Video representation. Safe to call
// concurrently with the worker writing progress.
func (j *Job) ToVideo() openai.Video {
	j.mu.RLock()
	defer j.mu.RUnlock()

	v := openai.Video{
		ID:                 j.id,
		Object:             openai.VideoObject,
		CreatedAt:          j.createdAt,
		Model:              j.model,
		Status:             j.status,
		Progress:           j.progress,
		Seconds:            j.seconds,
		Size:               j.size,
		RemixedFromVideoID: j.remixedFromID,
		Error:              j.err,
	}
	if j.prompt != "" {
		v.Prompt = j.prompt
	}
	if j.completedAt > 0 {
		v.CompletedAt = j.completedAt
	}
	if j.expiresAt > 0 {
		v.ExpiresAt = j.expiresAt
	}
	return v
}

// Content returns the stored MP4 bytes and content type for a completed job,
// or nil if the job is not completed or has no content. The bytes are not
// copied; callers must not mutate them.
func (j *Job) Content() ([]byte, string) {
	j.mu.RLock()
	defer j.mu.RUnlock()
	return j.content, j.contentType
}

// CachedVariant returns the cached PNG for a /content variant ("thumbnail"
// or "spritesheet"), computing and caching it via fn on first access. fn is
// called at most once per variant per job (the source MP4 is immutable).
// The job mutex is held during computation to avoid duplicate concurrent
// ffmpeg runs; variant extraction is fast for the short clips this API
// produces.
func (j *Job) CachedVariant(variant string, fn func() ([]byte, error)) ([]byte, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	var slot *[]byte
	switch variant {
	case "thumbnail":
		slot = &j.thumbnail
	case "spritesheet":
		slot = &j.spritesheet
	default:
		return nil, fmt.Errorf("unknown variant %q", variant)
	}
	if *slot != nil {
		return *slot, nil
	}
	b, err := fn()
	if err != nil {
		return nil, err
	}
	*slot = b
	return b, nil
}

// Status returns the current job status without locking for the full ToVideo
// snapshot. Used by retrieve/content handlers for fast terminal checks.
func (j *Job) Status() string {
	j.mu.RLock()
	defer j.mu.RUnlock()
	return j.status
}

// ID returns the job id.
func (j *Job) ID() string {
	return j.id
}

// ErrJobNotFound is returned by Get/Delete when no job exists for the id.
var ErrJobNotFound = errors.New("video job not found")

// newJobID generates a spec-plausible video job id: "vid_" + 24 hex chars.
func newJobID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "vid_" + hex.EncodeToString(b)
}

// JobStore is the interface for creating, retrieving, listing, and deleting
// video jobs. The in-memory implementation (memStore) is the default; a
// persistent implementation can satisfy this interface in the future.
type JobStore interface {
	Create(params CreateParams) (*Job, error)
	Get(id string) (*Job, error)
	Delete(id string) (bool, error)
	List(after string, limit int, order string) ([]*Job, bool)
	// Transcoder exposes the store's transcoder so HTTP handlers can run
	// on-demand operations (thumbnail/spritesheet variants, source-duration
	// probes) without re-resolving ffmpeg.
	Transcoder() Transcoder
	Close()
}

// memStore is the in-memory JobStore. Jobs are keyed by id; a background
// goroutine evicts expired jobs. A semaphore bounds concurrent generation.
type memStore struct {
	mu         sync.Mutex
	jobs       map[string]*Job
	order      []string // insertion order, for stable listing
	seqno      int64    // monotonic creation counter, for stable pagination
	sem        chan struct{}
	closed     chan struct{}
	transcoder Transcoder
}

// Transcoder converts a sequence of PNG-encoded frames into an MP4 container
// and back. It is injected so the package does not import x/diffgen (which is
// behind the sdcpp build tag). The default implementation shells out to
// ffmpeg.
type Transcoder interface {
	// EncodeMP4 transcodes the given PNG frames (in order) to a single MP4
	// container at fps. Returns the MP4 bytes or an error.
	EncodeMP4(ctx context.Context, framePNGs [][]byte, fps int) ([]byte, error)
	// DecodeFrames extracts PNG-encoded frames from an MP4/MKV video, starting
	// from the BEGINNING. maxFrames <= 0 means all frames (capped by
	// maxDecodedFrames). Used by /v1/videos/edits (first frame as I2V init).
	// The returned int is the probed fps (0 if unknown).
	DecodeFrames(ctx context.Context, mp4 []byte, maxFrames int) ([][]byte, int, error)
	// DecodeLastFrame extracts only the LAST frame of a video as a PNG. This
	// is used by /v1/videos/extensions, which continues the scene from the
	// source's final frame. It uses ffmpeg's -sseof (seek from end) so it does
	// not decode the entire source, avoiding the O(n) cost and the
	// first-N-frames truncation bug that DecodeFrames would hit for long
	// sources.
	DecodeLastFrame(ctx context.Context, mp4 []byte) ([]byte, error)
	// ConcatMP4 concatenates two MP4 byte streams (produced by EncodeMP4)
	// into a single MP4, preserving the fps. Used by /v1/videos/extensions to
	// stitch the source segment and the generated extension into one clip.
	// Either input may be nil/empty (the other is returned as-is).
	ConcatMP4(ctx context.Context, first, second []byte, fps int) ([]byte, error)
	// ProbeDurationSeconds returns the duration of a video in whole seconds
	// (rounded to nearest). Used by /v1/videos/extensions to report the
	// stitched total seconds when the source is an uploaded file (no job
	// record to read seconds from).
	ProbeDurationSeconds(ctx context.Context, mp4 []byte) (int, error)
	// Spritesheet renders a tiled grid of frames sampled across the video
	// as a single PNG (the spec's variant=spritesheet download).
	Spritesheet(ctx context.Context, mp4 []byte) ([]byte, error)
	// Available reports whether transcoding is possible (e.g. ffmpeg on PATH).
	Available() bool
}

// NewJobStore creates a new in-memory job store with the given transcoder
// and the default concurrency (MaxConcurrentJobs).
func NewJobStore(transcoder Transcoder) JobStore {
	return NewJobStoreWithConcurrency(transcoder, MaxConcurrentJobs)
}

// NewJobStoreWithConcurrency creates a new in-memory job store with the given
// transcoder and an explicit max-concurrent-jobs value (useful for env-driven
// configuration or tests that want parallelism). A concurrency of 0 is
// treated as 1 (at least one worker slot).
func NewJobStoreWithConcurrency(transcoder Transcoder, concurrency int) JobStore {
	if concurrency < 1 {
		concurrency = 1
	}
	s := &memStore{
		jobs:       make(map[string]*Job),
		sem:        make(chan struct{}, concurrency),
		closed:     make(chan struct{}),
		transcoder: transcoder,
	}
	go s.evictLoop()
	return s
}

// Create enqueues a new video job and starts a worker goroutine (subject to
// the concurrency semaphore). The returned Job has status "queued". The
// worker transitions it to "in_progress", then "completed" (with content) or
// "failed" (with error).
func (s *memStore) Create(params CreateParams) (*Job, error) {
	if params.Generate == nil {
		return nil, errors.New("videojobs: CreateParams.Generate is required")
	}
	now := time.Now().Unix()
	// Create the worker context up front and store its cancel func on the
	// job BEFORE launching the goroutine. This closes the DELETE race where
	// Delete sees j.cancel == nil between Create returning and run setting
	// it, which would leave an orphaned worker running after deletion. A
	// MaxJobAge timeout bounds hung runners so they can't leak indefinitely.
	ctx, cancel := context.WithTimeout(context.Background(), MaxJobAge)
	j := &Job{
		id:            newJobID(),
		model:         params.Model,
		prompt:        params.Prompt,
		seconds:       params.Seconds,
		size:          params.Size,
		remixedFromID: params.RemixedFromID,
		createdAt:     now,
		expiresAt:     0, // set on completion
		status:        openai.VideoStatusQueued,
		cancel:        cancel,
		done:          make(chan struct{}),
	}

	s.mu.Lock()
	s.seqno++
	j.seqno = s.seqno
	s.jobs[j.id] = j
	s.order = append(s.order, j.id)
	s.mu.Unlock()

	go s.run(j, params, ctx)
	return j, nil
}

// run acquires a worker slot, runs generation, transcodes to MP4, and sets
// the terminal status. Always closes j.done.
func (s *memStore) run(j *Job, params CreateParams, ctx context.Context) {
	defer close(j.done)
	defer j.cancelCtx()

	// Check transcoder availability up front for a fast, clear failure.
	if s.transcoder == nil || !s.transcoder.Available() {
		j.fail(&openai.VideoError{
			Code:    "ffmpeg_required",
			Message: "ffmpeg is required on PATH to produce MP4 content for the OpenAI Videos API",
		})
		return
	}

	// Acquire a concurrency slot (blocks if at capacity; the job stays
	// "queued" while waiting, which is spec-compliant).
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-s.closed:
		j.fail(&openai.VideoError{Code: "server_shutting_down", Message: "the server is shutting down"})
		return
	case <-ctx.Done():
		j.fail(&openai.VideoError{Code: "cancelled", Message: "job cancelled"})
		return
	}

	// Re-check map membership: Delete may have removed the job while we
	// waited for the semaphore. If so, exit without running.
	if _, ok := s.getJob(j.id); !ok {
		return
	}

	j.mu.Lock()
	j.status = openai.VideoStatusInProgress
	j.mu.Unlock()

	// For edit/extension jobs, derive the init image from the source video
	// before generation. Edits use the first frame (I2V init); extensions use
	// the last frame (continuation point). These are decoded via separate
	// ffmpeg calls so we don't hold the whole source in memory.
	var sourceMP4 []byte
	var sourceFps int
	if params.SourceVideoID != "" || len(params.SourceVideo) > 0 {
		var err error
		sourceMP4, sourceFps, err = s.resolveSourceVideo(ctx, params)
		if err != nil {
			j.fail(&openai.VideoError{Code: "source_video_unavailable", Message: err.Error()})
			return
		}
		if params.Extend {
			// Extensions continue from the LAST frame. Use DecodeLastFrame
			// (ffmpeg -sseof) which seeks to the end and decodes only the
			// final frame — correct for any source length and O(1) vs
			// DecodeFrames' O(n) + first-N-truncation bug.
			lastFrame, err := s.transcoder.DecodeLastFrame(ctx, sourceMP4)
			if err != nil {
				j.fail(&openai.VideoError{Code: "source_decode_failed", Message: fmt.Sprintf("failed to decode last frame: %v", err)})
				return
			}
			if len(lastFrame) == 0 {
				j.fail(&openai.VideoError{Code: "source_decode_failed", Message: "source video produced no frames"})
				return
			}
			params.InitImage = lastFrame
		} else {
			// Edits re-render from the FIRST frame as an I2V init image.
			frames, _, err := s.transcoder.DecodeFrames(ctx, sourceMP4, 1)
			if err != nil {
				j.fail(&openai.VideoError{Code: "source_decode_failed", Message: fmt.Sprintf("failed to decode source video: %v", err)})
				return
			}
			if len(frames) == 0 {
				j.fail(&openai.VideoError{Code: "source_decode_failed", Message: "source video produced no frames"})
				return
			}
			params.InitImage = frames[0]
		}
		// Free the source MP4 bytes now that frames are extracted, so the
		// large slice is GC-eligible before the (slow) generation + concat.
		// The extension still needs it for ConcatMP4, so only nil it for edits.
		if !params.Extend {
			sourceMP4 = nil
		}
	}

	var framePNGs [][]byte
	var lastStep, lastTotal int
	err := params.Generate(ctx, params, func(framePNG []byte, step, total int) {
		if len(framePNG) > 0 {
			// The runner's EncodeImageBase64 allocates a fresh buffer per
			// frame, so no defensive copy is needed.
			framePNGs = append(framePNGs, framePNG)
		}
		if total > 0 {
			lastStep, lastTotal = step, total
			pct := 0
			if total > 0 {
				pct = step * 100 / total
				if pct > 99 {
					pct = 99 // reserve 100 for completed
				}
			}
			j.mu.Lock()
			j.progress = pct
			j.mu.Unlock()
		}
	})
	if err != nil {
		j.fail(&openai.VideoError{Code: "generation_failed", Message: err.Error()})
		return
	}

	if len(framePNGs) == 0 {
		j.fail(&openai.VideoError{Code: "no_frames", Message: "generation produced no frames"})
		return
	}

	fps := secondsToFPS(j.seconds, lastStep, lastTotal)
	mp4, err := s.transcoder.EncodeMP4(ctx, framePNGs, fps)
	if err != nil {
		j.fail(&openai.VideoError{Code: "encoding_failed", Message: fmt.Sprintf("MP4 encoding failed: %v", err)})
		return
	}

	// For extensions, stitch source + generated into a single clip and
	// update the job's seconds to the stitched total.
	if params.Extend && len(sourceMP4) > 0 {
		concatFps := fps
		if sourceFps > 0 {
			concatFps = sourceFps
		}
		stitched, err := s.transcoder.ConcatMP4(ctx, sourceMP4, mp4, concatFps)
		if err != nil {
			j.fail(&openai.VideoError{Code: "concat_failed", Message: fmt.Sprintf("failed to stitch extension: %v", err)})
			return
		}
		mp4 = stitched
		j.setSeconds(stitchSeconds(params.SourceSeconds, params.Seconds))
	}

	now := time.Now().Unix()
	j.mu.Lock()
	j.status = openai.VideoStatusCompleted
	j.progress = 100
	j.content = mp4
	j.contentType = "video/mp4"
	j.completedAt = now
	j.expiresAt = now + int64(JobTTL.Seconds())
	j.mu.Unlock()

	// Enforce the global memory bound: if adding this content pushes total
	// retained bytes over the cap, evict the oldest completed jobs.
	s.evictForMemoryBudget()
}

// resolveSourceVideo returns the MP4 bytes and probed fps for an edit/extend
// job's source video. For SourceVideoID, it looks up the referenced completed
// job's content. For SourceVideo (an uploaded file), it returns the bytes
// directly with fps 0 (the caller falls back to the requested fps).
func (s *memStore) resolveSourceVideo(ctx context.Context, params CreateParams) ([]byte, int, error) {
	if len(params.SourceVideo) > 0 {
		return params.SourceVideo, 0, nil
	}
	if params.SourceVideoID != "" {
		src, err := s.Get(params.SourceVideoID)
		if err != nil {
			return nil, 0, fmt.Errorf("source video %q not found", params.SourceVideoID)
		}
		if src.Status() != openai.VideoStatusCompleted {
			return nil, 0, fmt.Errorf("source video %q is not completed (status: %s)", params.SourceVideoID, src.Status())
		}
		content, _ := src.Content()
		if len(content) == 0 {
			return nil, 0, fmt.Errorf("source video %q has no content", params.SourceVideoID)
		}
		return content, 0, nil
	}
	return nil, 0, errors.New("no source video provided")
}

// setSeconds updates the job's reported seconds (used for extension stitching).
func (j *Job) setSeconds(s string) {
	j.mu.Lock()
	j.seconds = s
	j.mu.Unlock()
}

// stitchSeconds computes the total seconds for an extension by adding the
// source seconds to the requested extension seconds. Both are spec strings
// ("4", "8", "12", ...). Non-numeric/empty values fall back to the requested
// seconds alone. The result is the decimal sum as a string (e.g. "8" + "4" =
// "12").
func stitchSeconds(source, requested string) string {
	src, err1 := strconv.Atoi(source)
	req, err2 := strconv.Atoi(requested)
	if err1 != nil || err2 != nil || src <= 0 {
		return requested
	}
	return strconv.Itoa(src + req)
}

// getJob returns the job for the given id under the store lock.
func (s *memStore) getJob(id string) (*Job, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	return j, ok
}

// fail transitions the job to the failed status with the given error.
func (j *Job) fail(e *openai.VideoError) {
	now := time.Now().Unix()
	j.mu.Lock()
	j.status = openai.VideoStatusFailed
	j.err = e
	j.completedAt = now
	j.expiresAt = now + int64(JobTTL.Seconds())
	j.mu.Unlock()
}

// secondsToFPS derives an output fps from the requested seconds. The OpenAI
// spec only exposes "seconds" (4/8/12), not fps; we use 16 fps (WAN default)
// unless a different signal is available. step/total are passed for future
// refinement but not currently used.
func secondsToFPS(seconds string, step, total int) int {
	return 16
}

// Get returns the job for the given id, or ErrJobNotFound.
func (s *memStore) Get(id string) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return nil, ErrJobNotFound
	}
	return j, nil
}

// Delete removes a job, cancels its worker if still running, and reports
// whether a job was deleted. Per the spec, DELETE returns deleted=true even
// for in-flight jobs (the job is cancelled and removed).
func (s *memStore) Delete(id string) (bool, error) {
	s.mu.Lock()
	j, ok := s.jobs[id]
	if !ok {
		s.mu.Unlock()
		return false, ErrJobNotFound
	}
	delete(s.jobs, id)
	s.order = removeString(s.order, id)
	s.mu.Unlock()

	// Cancel outside the store lock to avoid holding it during any
	// cancellation side effects. cancelCtx is a no-op if already cancelled.
	j.cancelCtx()
	// Clear content immediately to free memory.
	j.mu.Lock()
	j.content = nil
	j.contentType = ""
	j.thumbnail = nil
	j.spritesheet = nil
	j.mu.Unlock()
	return true, nil
}

// List returns up to limit jobs in the requested order ("asc" or "desc"),
// starting after the given cursor id (empty = from the start). Returns the
// slice and whether more results exist (has_more). The cursor pagination
// follows the OpenAI convention: after is a job id; results are strictly
// after it in the chosen order. If the cursor job has been deleted/evicted
// between calls (cursor not found among current entries), the filter is
// disabled and all entries are returned (degraded to "restart from start")
// rather than returning an empty page — this avoids data loss on the desc
// path (which would otherwise skip every entry because cursorSeq stays -1).
func (s *memStore) List(after string, limit int, order string) ([]*Job, bool) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	if order == "" {
		order = "desc"
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Build a snapshot of (id, seqno) sorted by seqno ascending (oldest first).
	type entry struct {
		id  string
		seq int64
	}
	entries := make([]entry, 0, len(s.order))
	for _, id := range s.order {
		if j, ok := s.jobs[id]; ok {
			entries = append(entries, entry{id: id, seq: j.seqno})
		}
	}
	// Sort by seqno ascending (oldest first). Since s.order is insertion
	// order and seqno is assigned at insertion, this is already sorted, but
	// sort explicitly to be robust against map deletion/reinsertion.
	for i := 0; i < len(entries); i++ {
		min := i
		for k := i + 1; k < len(entries); k++ {
			if entries[k].seq < entries[min].seq {
				min = k
			}
		}
		entries[i], entries[min] = entries[min], entries[i]
	}

	// Determine the cursor seqno. If `after` matches an existing job, use
	// its seqno to filter strictly-after results. If the job was
	// deleted/evicted (not found), disable the filter so pagination
	// degrades to returning all entries rather than an empty page.
	cursorSeq := int64(-1)
	cursorFound := false
	if after != "" {
		for _, e := range entries {
			if e.id == after {
				cursorSeq = e.seq
				cursorFound = true
				break
			}
		}
	}

	// For desc order, iterate entries in reverse (newest first).
	var filtered []*Job
	if order == "asc" {
		for _, e := range entries {
			if cursorFound && e.seq <= cursorSeq {
				continue
			}
			if j, ok := s.jobs[e.id]; ok {
				filtered = append(filtered, j)
			}
		}
	} else { // desc
		for i := len(entries) - 1; i >= 0; i-- {
			e := entries[i]
			if cursorFound && e.seq >= cursorSeq {
				continue
			}
			if j, ok := s.jobs[e.id]; ok {
				filtered = append(filtered, j)
			}
		}
	}

	// Apply limit.
	if len(filtered) <= limit {
		return filtered, false
	}
	return filtered[:limit], true
}

// Transcoder returns the store's transcoder.
func (s *memStore) Transcoder() Transcoder {
	return s.transcoder
}

// Close cancels all in-flight jobs and stops the eviction loop.
func (s *memStore) Close() {
	close(s.closed)
	s.mu.Lock()
	for _, j := range s.jobs {
		j.mu.RLock()
		terminal := j.status == openai.VideoStatusCompleted || j.status == openai.VideoStatusFailed
		j.mu.RUnlock()
		if !terminal {
			j.cancelCtx()
		}
	}
	s.mu.Unlock()
}

// evictLoop periodically removes expired jobs to bound memory usage.
func (s *memStore) evictLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-s.closed:
			return
		case <-ticker.C:
			s.evictExpired()
		}
	}
}

func (s *memStore) evictExpired() {
	now := time.Now().Unix()
	maxAgeUnix := int64(MaxJobAge.Seconds())
	toDelete := []string{}
	s.mu.Lock()
	for id, j := range s.jobs {
		j.mu.RLock()
		exp := j.expiresAt
		status := j.status
		created := j.createdAt
		terminal := status == openai.VideoStatusCompleted || status == openai.VideoStatusFailed
		j.mu.RUnlock()

		// Evict terminal jobs past their TTL.
		if terminal && exp > 0 && now > exp {
			toDelete = append(toDelete, id)
			continue
		}
		// Force-fail non-terminal jobs that have exceeded MaxJobAge
		// (hung runner / leaked concurrency slot).
		if !terminal && created > 0 && (now-created) > maxAgeUnix {
			j.fail(&openai.VideoError{
				Code:    "timeout",
				Message: fmt.Sprintf("job exceeded the maximum age of %s", MaxJobAge),
			})
			// Mark for eviction after the TTL set by fail().
			toDelete = append(toDelete, id)
		}
	}
	for _, id := range toDelete {
		delete(s.jobs, id)
		s.order = removeString(s.order, id)
	}
	s.mu.Unlock()

	// Enforce the global memory budget after TTL/max-age eviction.
	s.evictForMemoryBudget()
}

// evictForMemoryBudget evicts the oldest completed jobs (by createdAt) until
// the total retained content bytes is under MaxTotalContentBytes. Skips
// non-completed jobs (their content is nil anyway).
func (s *memStore) evictForMemoryBudget() {
	s.mu.Lock()
	defer s.mu.Unlock()

	var total int64
	for _, j := range s.jobs {
		j.mu.RLock()
		if j.status == openai.VideoStatusCompleted {
			total += int64(len(j.content))
		}
		j.mu.RUnlock()
	}

	if total <= MaxTotalContentBytes {
		return
	}

	// Build a list of completed jobs sorted by seqno ascending (oldest
	// first), evict until under budget.
	type cand struct {
		id   string
		seq  int64
		size int
	}
	var cands []cand
	for id, j := range s.jobs {
		j.mu.RLock()
		if j.status == openai.VideoStatusCompleted {
			cands = append(cands, cand{id: id, seq: j.seqno, size: len(j.content)})
		}
		j.mu.RUnlock()
	}
	// Selection sort (N is small): oldest first.
	for i := 0; i < len(cands); i++ {
		min := i
		for k := i + 1; k < len(cands); k++ {
			if cands[k].seq < cands[min].seq {
				min = k
			}
		}
		cands[i], cands[min] = cands[min], cands[i]
	}

	for _, c := range cands {
		if total <= MaxTotalContentBytes {
			break
		}
		delete(s.jobs, c.id)
		s.order = removeString(s.order, c.id)
		total -= int64(c.size)
	}
}

func removeString(s []string, v string) []string {
	for i, x := range s {
		if x == v {
			return append(s[:i], s[i+1:]...)
		}
	}
	return s
}
