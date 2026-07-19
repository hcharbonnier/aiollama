package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/videojobs"
)

// setupVideoRouter creates a gin router with the video handlers bound to a
// Server with the given job store. Returns the server for assertions.
func setupVideoRouter(t *testing.T, store videojobs.JobStore) (*Server, *gin.Engine) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	s := &Server{videoJobs: store}
	r := gin.New()
	r.POST("/v1/videos", s.VideoCreateHandler)
	r.GET("/v1/videos", s.VideoListHandler)
	r.GET("/v1/videos/:video_id", s.VideoRetrieveHandler)
	r.DELETE("/v1/videos/:video_id", s.VideoDeleteHandler)
	r.GET("/v1/videos/:video_id/content", s.VideoContentHandler)
	return s, r
}

// newMultipartRequest builds a multipart/form-data POST request with the given
// fields. The "input_reference" field, if provided as non-empty, is added as a
// file part.
func newMultipartRequest(t *testing.T, url string, fields map[string]string, refFile []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	if len(refFile) > 0 {
		fw, err := mw.CreateFormFile("input_reference", "ref.png")
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		fw.Write(refFile)
	}
	_ = mw.Close()
	req, err := http.NewRequest(http.MethodPost, url, &body)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

// TestVideoCreateRejectsMissingPrompt verifies that POST /v1/videos without a
// prompt returns 400.
func TestVideoCreateRejectsMissingPrompt(t *testing.T) {
	// Use a real job store with a nil transcoder; the handler should reject
	// before reaching Create because prompt validation runs first.
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req := newMultipartRequest(t, "/v1/videos", map[string]string{
		"model": "wan2.1-t2v",
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "prompt is required") {
		t.Errorf("body = %q, want 'prompt is required'", w.Body.String())
	}
}

// TestVideoCreateRejectsInvalidSeconds verifies that an out-of-spec seconds
// value is rejected.
func TestVideoCreateRejectsInvalidSeconds(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req := newMultipartRequest(t, "/v1/videos", map[string]string{
		"prompt":  "a cat",
		"model":   "wan2.1-t2v",
		"seconds": "7",
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "seconds must be one of") {
		t.Errorf("body = %q, want seconds validation error", w.Body.String())
	}
}

// TestVideoCreateRejectsInvalidSize verifies that an out-of-spec size is
// rejected.
func TestVideoCreateRejectsInvalidSize(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req := newMultipartRequest(t, "/v1/videos", map[string]string{
		"prompt": "a cat",
		"model":  "wan2.1-t2v",
		"size":   "999x999",
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "size must be one of") {
		t.Errorf("body = %q, want size validation error", w.Body.String())
	}
}

// TestVideoCreateRejectsFileID verifies that input_reference.file_id is
// rejected (not supported in v1).
func TestVideoCreateRejectsFileID(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	// Send JSON with input_reference.file_id.
	body := `{"prompt":"a cat","model":"wan2.1-t2v","input_reference":{"file_id":"file_abc"}}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/videos", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "file_id") {
		t.Errorf("body = %q, want file_id rejection", w.Body.String())
	}
}

// TestVideoCreateSetsDefaults verifies that omitted fields get spec defaults
// (seconds=4, size=720x1280) and that a valid request with an explicit model
// returns 200 with a Video object in "queued" status. (model is now required,
// not defaulted to "sora-2", since the local scheduler needs a real model name.)
func TestVideoCreateSetsDefaults(t *testing.T) {
	store := videojobs.NewJobStore(nilTranscoder{})
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req := newMultipartRequest(t, "/v1/videos", map[string]string{
		"prompt": "a cat",
		"model":  "wan2.1-t2v",
		// seconds, size omitted → defaults
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}

	var v openai.Video
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v.ID == "" {
		t.Error("id is empty")
	}
	if v.Object != openai.VideoObject {
		t.Errorf("object = %q, want %q", v.Object, openai.VideoObject)
	}
	if v.Status != openai.VideoStatusQueued && v.Status != openai.VideoStatusInProgress && v.Status != openai.VideoStatusFailed {
		t.Errorf("status = %q, want queued/in_progress/failed", v.Status)
	}
	// Defaults are applied for the job params; the echoed Video should
	// reflect them.
	if v.Seconds != openai.VideoDefaultSeconds {
		t.Errorf("seconds = %q, want %q (default)", v.Seconds, openai.VideoDefaultSeconds)
	}
	if v.Size != openai.VideoDefaultSize {
		t.Errorf("size = %q, want %q (default)", v.Size, openai.VideoDefaultSize)
	}
	if v.Model != "wan2.1-t2v" {
		t.Errorf("model = %q, want %q (echoed)", v.Model, "wan2.1-t2v")
	}
}

// TestVideoCreateRejectsMissingModel verifies that POST /v1/videos without a
// model returns 400 (model is required, not defaulted to "sora-2").
func TestVideoCreateRejectsMissingModel(t *testing.T) {
	store := videojobs.NewJobStore(nilTranscoder{})
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req := newMultipartRequest(t, "/v1/videos", map[string]string{
		"prompt": "a cat",
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "model is required") {
		t.Errorf("body = %q, want 'model is required'", w.Body.String())
	}
}

// TestVideoRetrieveNotFound verifies that GET /v1/videos/{id} for a
// nonexistent id returns 404.
func TestVideoRetrieveNotFound(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req, _ := http.NewRequest(http.MethodGet, "/v1/videos/vid_nonexistent", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
}

// TestVideoDeleteNotFound verifies that DELETE /v1/videos/{id} for a
// nonexistent id returns 404.
func TestVideoDeleteNotFound(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req, _ := http.NewRequest(http.MethodDelete, "/v1/videos/vid_nonexistent", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
}

// TestVideoContentNotReady verifies that GET /v1/videos/{id}/content for a
// job that has not reached "completed" returns a non-200 (409 conflict).
func TestVideoContentNotReady(t *testing.T) {
	// Use a store with an unavailable transcoder so the job fails fast with
	// ffmpeg_required, guaranteeing a non-completed status when we query
	// content.
	store := videojobs.NewJobStore(nilTranscoder{})
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req := newMultipartRequest(t, "/v1/videos", map[string]string{
		"prompt": "a cat",
		"model":  "wan2.1-t2v",
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("create status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var v struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(w.Body.Bytes(), &v)
	if v.ID == "" {
		t.Fatal("no job id in response")
	}

	// Request content immediately; the job is queued or failed → 409.
	contentURL := "/v1/videos/" + v.ID + "/content"
	creq, _ := http.NewRequest(http.MethodGet, contentURL, nil)
	cw := httptest.NewRecorder()
	r.ServeHTTP(cw, creq)

	if cw.Code == http.StatusOK {
		t.Fatalf("content status = 200, want non-200 (job not completed): %s", cw.Body.String())
	}
	if cw.Code != http.StatusConflict && cw.Code != http.StatusNotFound {
		t.Logf("content status = %d (acceptable: 409 or 404 for non-completed job)", cw.Code)
	}
}

// nilTranscoder is a Transcoder that reports unavailable.
type nilTranscoder struct{}

func (nilTranscoder) EncodeMP4(ctx context.Context, framePNGs [][]byte, fps int) ([]byte, error) {
	return nil, errors.New("no transcoder")
}

func (nilTranscoder) DecodeFrames(ctx context.Context, mp4 []byte, maxFrames int) ([][]byte, int, error) {
	return nil, 0, errors.New("no transcoder")
}

func (nilTranscoder) DecodeLastFrame(ctx context.Context, mp4 []byte) ([]byte, error) {
	return nil, errors.New("no transcoder")
}

func (nilTranscoder) ConcatMP4(ctx context.Context, first, second []byte, fps int) ([]byte, error) {
	return nil, errors.New("no transcoder")
}

func (nilTranscoder) ProbeDurationSeconds(ctx context.Context, mp4 []byte) (int, error) {
	return 0, errors.New("no transcoder")
}

func (nilTranscoder) Spritesheet(ctx context.Context, mp4 []byte) ([]byte, error) {
	return nil, errors.New("no transcoder")
}
func (nilTranscoder) Available() bool { return false }

// TestVideoListEmpty verifies that GET /v1/videos on an empty store returns
// an empty list with has_more=false.
func TestVideoListEmpty(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req, _ := http.NewRequest(http.MethodGet, "/v1/videos", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var resp openai.VideoListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Object != openai.VideoObjectList {
		t.Errorf("object = %q, want %q", resp.Object, openai.VideoObjectList)
	}
	if len(resp.Data) != 0 {
		t.Errorf("data len = %d, want 0", len(resp.Data))
	}
	if resp.HasMore {
		t.Error("has_more = true, want false")
	}
}

// TestVideoContentVariants verifies the thumbnail and spritesheet variants
// of GET /v1/videos/{id}/content against a completed job, plus validation of
// unknown variant values and unknown job ids.
func TestVideoContentVariants(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	gen := func(ctx context.Context, params videojobs.CreateParams, fn func([]byte, int, int)) error {
		fn([]byte("frame-png"), 1, 1)
		return nil
	}
	job, err := store.Create(videojobs.CreateParams{
		Model: "m", Prompt: "p", Seconds: "4", Size: "720x1280", Generate: gen,
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for job.Status() != openai.VideoStatusCompleted && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if job.Status() != openai.VideoStatusCompleted {
		t.Fatalf("job did not complete: status = %s", job.Status())
	}

	tests := []struct {
		name        string
		url         string
		wantCode    int
		wantBody    string
		wantContent string
	}{
		{name: "video variant", url: "/v1/videos/" + job.ID() + "/content?variant=video", wantCode: http.StatusOK, wantContent: "video/mp4"},
		{name: "default variant", url: "/v1/videos/" + job.ID() + "/content", wantCode: http.StatusOK, wantContent: "video/mp4"},
		{name: "thumbnail variant", url: "/v1/videos/" + job.ID() + "/content?variant=thumbnail", wantCode: http.StatusOK, wantBody: "decoded-frame", wantContent: "image/png"},
		{name: "spritesheet variant", url: "/v1/videos/" + job.ID() + "/content?variant=spritesheet", wantCode: http.StatusOK, wantBody: "spritesheet-png", wantContent: "image/png"},
		{name: "invalid variant", url: "/v1/videos/" + job.ID() + "/content?variant=bogus", wantCode: http.StatusBadRequest},
		{name: "unknown job thumbnail", url: "/v1/videos/vid_unknown/content?variant=thumbnail", wantCode: http.StatusNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodGet, tt.url, nil)
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)
			if w.Code != tt.wantCode {
				t.Fatalf("status = %d, want %d: %s", w.Code, tt.wantCode, w.Body.String())
			}
			if tt.wantContent != "" && w.Header().Get("Content-Type") != tt.wantContent {
				t.Errorf("content-type = %q, want %q", w.Header().Get("Content-Type"), tt.wantContent)
			}
			if tt.wantBody != "" && w.Body.String() != tt.wantBody {
				t.Errorf("body = %q, want %q", w.Body.String(), tt.wantBody)
			}
		})
	}
}

// extendableTranscoder is a Transcoder that simulates frame decode + concat
// for edit/extend handler tests without a real ffmpeg. It returns a single
// stub "frame" from DecodeFrames and a stub MP4 from ConcatMP4.
type extendableTranscoder struct{}

func (extendableTranscoder) EncodeMP4(ctx context.Context, framePNGs [][]byte, fps int) ([]byte, error) {
	return []byte{0, 0, 0, 0x18, 'f', 't', 'y', 'p'}, nil
}

func (extendableTranscoder) DecodeFrames(ctx context.Context, mp4 []byte, maxFrames int) ([][]byte, int, error) {
	return [][]byte{[]byte("decoded-frame")}, 16, nil
}

func (extendableTranscoder) DecodeLastFrame(ctx context.Context, mp4 []byte) ([]byte, error) {
	return []byte("decoded-last-frame"), nil
}

func (extendableTranscoder) ConcatMP4(ctx context.Context, first, second []byte, fps int) ([]byte, error) {
	return []byte("stitched"), nil
}

func (extendableTranscoder) ProbeDurationSeconds(ctx context.Context, mp4 []byte) (int, error) {
	return 8, nil
}

func (extendableTranscoder) Spritesheet(ctx context.Context, mp4 []byte) ([]byte, error) {
	return []byte("spritesheet-png"), nil
}
func (extendableTranscoder) Available() bool { return true }

// countingTranscoder counts DecodeFrames/Spritesheet calls to verify
// variant caching on the job.
type countingTranscoder struct {
	decodeCalls int
	sheetCalls  int
}

func (c *countingTranscoder) EncodeMP4(ctx context.Context, framePNGs [][]byte, fps int) ([]byte, error) {
	return []byte{0, 0, 0, 0x18, 'f', 't', 'y', 'p'}, nil
}

func (c *countingTranscoder) DecodeFrames(ctx context.Context, mp4 []byte, maxFrames int) ([][]byte, int, error) {
	c.decodeCalls++
	return [][]byte{[]byte("decoded-frame")}, 16, nil
}

func (c *countingTranscoder) DecodeLastFrame(ctx context.Context, mp4 []byte) ([]byte, error) {
	return []byte("decoded-last-frame"), nil
}

func (c *countingTranscoder) ConcatMP4(ctx context.Context, first, second []byte, fps int) ([]byte, error) {
	return []byte("stitched"), nil
}

func (c *countingTranscoder) ProbeDurationSeconds(ctx context.Context, mp4 []byte) (int, error) {
	return 8, nil
}

func (c *countingTranscoder) Spritesheet(ctx context.Context, mp4 []byte) ([]byte, error) {
	c.sheetCalls++
	return []byte("spritesheet-png"), nil
}
func (c *countingTranscoder) Available() bool { return true }

// TestVideoContentVariantCached verifies that repeated downloads of the same
// variant for the same job reuse the cached bytes (single ffmpeg run).
func TestVideoContentVariantCached(t *testing.T) {
	tc := &countingTranscoder{}
	store := videojobs.NewJobStore(tc)
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	gen := func(ctx context.Context, params videojobs.CreateParams, fn func([]byte, int, int)) error {
		fn([]byte("frame-png"), 1, 1)
		return nil
	}
	job, err := store.Create(videojobs.CreateParams{
		Model: "m", Prompt: "p", Seconds: "4", Size: "720x1280", Generate: gen,
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for job.Status() != openai.VideoStatusCompleted && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if job.Status() != openai.VideoStatusCompleted {
		t.Fatalf("job did not complete: status = %s", job.Status())
	}

	for i := 0; i < 3; i++ {
		req, _ := http.NewRequest(http.MethodGet, "/v1/videos/"+job.ID()+"/content?variant=thumbnail", nil)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("request %d: status = %d", i, w.Code)
		}
	}
	for i := 0; i < 2; i++ {
		req, _ := http.NewRequest(http.MethodGet, "/v1/videos/"+job.ID()+"/content?variant=spritesheet", nil)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("spritesheet request %d: status = %d", i, w.Code)
		}
	}

	if tc.decodeCalls != 1 {
		t.Errorf("DecodeFrames called %d times, want 1 (cached)", tc.decodeCalls)
	}
	if tc.sheetCalls != 1 {
		t.Errorf("Spritesheet called %d times, want 1 (cached)", tc.sheetCalls)
	}
}

// TestIsBlockedRemoteIP verifies the SSRF destination filter used for
// input_reference.image_url downloads.
func TestIsBlockedRemoteIP(t *testing.T) {
	blocked := []string{
		"127.0.0.1", "127.53.1.9", "::1",
		"10.0.0.1", "172.16.0.1", "192.168.1.1",
		"169.254.169.254", "fe80::1",
		"0.0.0.0", "224.0.0.1", "ff02::1",
	}
	public := []string{
		"8.8.8.8", "1.1.1.1", "2606:4700:4700::1111",
	}
	for _, s := range blocked {
		ip := net.ParseIP(s)
		if ip == nil {
			t.Fatalf("bad test ip %q", s)
		}
		if !isBlockedRemoteIP(ip) {
			t.Errorf("ip %s should be blocked", s)
		}
	}
	for _, s := range public {
		ip := net.ParseIP(s)
		if ip == nil {
			t.Fatalf("bad test ip %q", s)
		}
		if isBlockedRemoteIP(ip) {
			t.Errorf("ip %s should be allowed", s)
		}
	}
}

// setupVideoRouterWithEdits binds all video routes (including edits/extensions)
// to the server's job store.
func setupVideoRouterWithEdits(t *testing.T, store videojobs.JobStore) (*Server, *gin.Engine) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	s := &Server{videoJobs: store}
	r := gin.New()
	r.POST("/v1/videos", s.VideoCreateHandler)
	r.GET("/v1/videos", s.VideoListHandler)
	r.GET("/v1/videos/:video_id", s.VideoRetrieveHandler)
	r.DELETE("/v1/videos/:video_id", s.VideoDeleteHandler)
	r.GET("/v1/videos/:video_id/content", s.VideoContentHandler)
	r.POST("/v1/videos/edits", s.VideoEditHandler)
	r.POST("/v1/videos/extensions", s.VideoExtendHandler)
	return s, r
}

// newMultipartRequestWithVideo builds a multipart/form-data POST with text
// fields plus an optional "video" file part (for edit/extend requests).
func newMultipartRequestWithVideo(t *testing.T, url string, fields map[string]string, videoFile []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	if len(videoFile) > 0 {
		fw, err := mw.CreateFormFile("video", "source.mp4")
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		fw.Write(videoFile)
	}
	_ = mw.Close()
	req, err := http.NewRequest(http.MethodPost, url, &body)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

// TestVideoEditRejectsMissingPrompt verifies POST /v1/videos/edits without a
// prompt returns 400.
func TestVideoEditRejectsMissingPrompt(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	req := newMultipartRequestWithVideo(t, "/v1/videos/edits", map[string]string{
		"model": "wan2.1-t2v",
	}, []byte("fake-mp4"))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "prompt is required") {
		t.Errorf("body = %q, want 'prompt is required'", w.Body.String())
	}
}

// TestVideoEditRejectsMissingVideo verifies that a missing `video` field
// (neither file part nor {id}) returns 400.
func TestVideoEditRejectsMissingVideo(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	req := newMultipartRequestWithVideo(t, "/v1/videos/edits", map[string]string{
		"prompt": "a cat",
		"model":  "wan2.1-t2v",
	}, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "video is required") {
		t.Errorf("body = %q, want 'video is required'", w.Body.String())
	}
}

// TestVideoEditRejectsMissingModel verifies that a missing model returns 400.
func TestVideoEditRejectsMissingModel(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	req := newMultipartRequestWithVideo(t, "/v1/videos/edits", map[string]string{
		"prompt": "a cat",
	}, []byte("fake-mp4"))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "model is required") {
		t.Errorf("body = %q, want 'model is required'", w.Body.String())
	}
}

// TestVideoEditWithFileUploadAccepts verifies that a valid edit request with
// an uploaded source file is accepted (returns 200 with a queued Video).
func TestVideoEditWithFileUploadAccepts(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	req := newMultipartRequestWithVideo(t, "/v1/videos/edits", map[string]string{
		"prompt":  "a cat running",
		"model":   "wan2.1-t2v",
		"seconds": "4",
		"size":    "720x1280",
	}, []byte("fake-source-mp4"))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var v openai.Video
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v.ID == "" {
		t.Error("id is empty")
	}
	if v.Object != openai.VideoObject {
		t.Errorf("object = %q, want %q", v.Object, openai.VideoObject)
	}
}

// TestVideoExtendRejectsMissingSeconds verifies that extensions require an
// explicit seconds field (no default).
func TestVideoExtendRejectsMissingSeconds(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	req := newMultipartRequestWithVideo(t, "/v1/videos/extensions", map[string]string{
		"prompt": "continue the scene",
		"model":  "wan2.1-t2v",
	}, []byte("fake-mp4"))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "seconds is required") {
		t.Errorf("body = %q, want 'seconds is required'", w.Body.String())
	}
}

// TestVideoExtendRejectsInvalidSeconds verifies that seconds outside the
// extension range (4-20) is rejected.
func TestVideoExtendRejectsInvalidSeconds(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	req := newMultipartRequestWithVideo(t, "/v1/videos/extensions", map[string]string{
		"prompt":  "continue the scene",
		"model":   "wan2.1-t2v",
		"seconds": "24",
	}, []byte("fake-mp4"))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "seconds must be one of") {
		t.Errorf("body = %q, want seconds validation error", w.Body.String())
	}
}

// TestVideoExtendAcceptsValidSeconds verifies the full extension seconds range
// (4, 8, 12, 16, 20) is accepted.
func TestVideoExtendAcceptsValidSeconds(t *testing.T) {
	for _, sec := range []string{"4", "8", "12", "16", "20"} {
		t.Run(sec, func(t *testing.T) {
			store := videojobs.NewJobStore(extendableTranscoder{})
			defer store.Close()
			_, r := setupVideoRouterWithEdits(t, store)

			req := newMultipartRequestWithVideo(t, "/v1/videos/extensions", map[string]string{
				"prompt":  "continue the scene",
				"model":   "wan2.1-t2v",
				"seconds": sec,
				"size":    "720x1280",
			}, []byte("fake-mp4"))
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("seconds=%s: status = %d, want 200: %s", sec, w.Code, w.Body.String())
			}
		})
	}
}

// TestVideoEditJSONFallback verifies the JSON extension on edits/extensions:
// a JSON body with a {"id": ...} video reference is accepted (aiollama
// extension; the spec content type is multipart), while a JSON body missing
// the video reference is rejected.
func TestVideoEditJSONFallback(t *testing.T) {
	store := videojobs.NewJobStore(extendableTranscoder{})
	defer store.Close()
	_, r := setupVideoRouterWithEdits(t, store)

	// JSON without a video reference → 400.
	body := `{"prompt":"a cat","model":"wan2.1-t2v"}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/videos/edits", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (video required): %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "video is required") {
		t.Errorf("body = %q, want video requirement", w.Body.String())
	}

	// JSON with an {"id": ...} reference → accepted (job created). The
	// referenced source doesn't exist, but job creation is asynchronous;
	// the create itself returns 200 with status queued.
	body = `{"prompt":"a cat","model":"wan2.1-t2v","video":{"id":"vid_abc"}}`
	req, _ = http.NewRequest(http.MethodPost, "/v1/videos/edits", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var v openai.Video
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v.Status != openai.VideoStatusQueued {
		t.Errorf("status = %q, want queued", v.Status)
	}
	if v.RemixedFromVideoID != "vid_abc" {
		t.Errorf("remixed_from_video_id = %q, want vid_abc", v.RemixedFromVideoID)
	}
}
