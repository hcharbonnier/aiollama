package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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

// TestVideoContentVariantNotImplemented verifies that variant=thumbnail
// returns 501.
func TestVideoContentVariantNotImplemented(t *testing.T) {
	store := videojobs.NewJobStore(nil)
	defer store.Close()
	_, r := setupVideoRouter(t, store)

	req, _ := http.NewRequest(http.MethodGet, "/v1/videos/vid_x/content?variant=thumbnail", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", w.Code)
	}
}
