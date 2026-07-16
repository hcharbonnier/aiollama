//go:build sdcpp

// Handler unit tests for the diffgen runner. These exercise the HTTP handlers
// (handleImageCompletion, handleVideoCompletion) with a mock sdContext so no
// GPU, model files, or real libstable-diffusion calls are needed. The test
// binary still links libstable-diffusion (because x/sdcpp is cgo), but the
// mock never calls into C. Run with:
//
//	go test -tags=sdcpp ./x/diffgen/ -run TestHandler
//
// See docs/video-generation-implementation-plan.md §12.2.

package diffgen

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/x/sdcpp"
)

// mockSDContext is a test double for sdContext. It records the params it
// receives and returns canned results. The progress callback is invoked
// synchronously to simulate SD.cpp step callbacks.
type mockSDContext struct {
	mu sync.Mutex

	supportsImage bool
	supportsVideo bool

	imageResult []sdcpp.Image
	imageErr    error

	videoResult []sdcpp.Image
	videoErr    error

	cancelled   bool
	cancelMu    sync.Mutex
	cancelCount int

	closed bool

	lastImageParams sdcpp.ImageGenParams
	lastVideoParams sdcpp.VideoGenParams
	progressSteps   int // number of progress callbacks to emit
}

func (m *mockSDContext) GenerateImage(p sdcpp.ImageGenParams, progress sdcpp.ProgressFunc) ([]sdcpp.Image, error) {
	m.mu.Lock()
	m.lastImageParams = p
	m.mu.Unlock()
	for i := 1; i <= m.progressSteps; i++ {
		if progress != nil {
			progress(i, m.progressSteps, 0)
		}
	}
	return m.imageResult, m.imageErr
}

func (m *mockSDContext) GenerateVideo(p sdcpp.VideoGenParams, progress sdcpp.ProgressFunc) ([]sdcpp.Image, error) {
	m.mu.Lock()
	m.lastVideoParams = p
	m.mu.Unlock()
	for i := 1; i <= m.progressSteps; i++ {
		if progress != nil {
			progress(i, m.progressSteps, 0)
		}
	}
	return m.videoResult, m.videoErr
}

func (m *mockSDContext) CancelGeneration(mode sdcpp.CancelMode) {
	m.cancelMu.Lock()
	m.cancelled = true
	m.cancelCount++
	m.cancelMu.Unlock()
}

func (m *mockSDContext) SupportsImageGeneration() bool { return m.supportsImage }
func (m *mockSDContext) SupportsVideoGeneration() bool { return m.supportsVideo }
func (m *mockSDContext) Close() {
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()
}

func (m *mockSDContext) wasCancelled() bool {
	m.cancelMu.Lock()
	defer m.cancelMu.Unlock()
	return m.cancelled
}

// testImage builds a small valid sdcpp.Image (1x1 RGB).
func testImage() sdcpp.Image {
	return sdcpp.Image{Width: 1, Height: 1, Channel: 3, Data: []byte{255, 0, 0}}
}

// testPNGBytes returns a valid 1x1 PNG image as bytes.
func testPNGBytes() []byte {
	var buf bytes.Buffer
	png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 1, 1)))
	return buf.Bytes()
}

// newTestServer builds a runnerServer wired to a mock context.
func newTestServer(mode ModelMode, mock *mockSDContext) *runnerServer {
	return &runnerServer{
		modelName: "test-model",
		port:      0,
		ctx:       mock,
		mode:      mode,
	}
}

// decodeNDJSON reads ndjson lines from the response body into []DiffResponse.
func decodeNDJSON(t *testing.T, body io.Reader) []DiffResponse {
	t.Helper()
	dec := json.NewDecoder(body)
	var resps []DiffResponse
	for {
		var r DiffResponse
		if err := dec.Decode(&r); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("decode ndjson: %v", err)
		}
		resps = append(resps, r)
	}
	return resps
}

func doCompletion(t *testing.T, s *runnerServer, req DiffRequest) (*http.Response, []DiffResponse) {
	t.Helper()
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	r := httptest.NewRequest(http.MethodPost, "/completion", bytes.NewReader(body))
	r = r.WithContext(context.Background())
	w := httptest.NewRecorder()
	s.completionHandler(w, r)
	resp := w.Result()
	resps := decodeNDJSON(t, resp.Body)
	return resp, resps
}

func TestHandlerImageStreaming(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
		progressSteps: 3,
	}
	s := newTestServer(ModeImage, mock)

	resp, resps := doCompletion(t, s, DiffRequest{
		Prompt: "a cat",
		Width:  64, Height: 64, Steps: 3, Seed: 42,
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if resp.Header.Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("content-type = %q", resp.Header.Get("Content-Type"))
	}

	var progressCount, doneCount, imageCount int
	for _, r := range resps {
		if r.Done {
			doneCount++
			if r.Image == "" {
				t.Errorf("done response missing image")
			} else {
				imageCount++
			}
		} else if r.Step > 0 {
			progressCount++
		}
	}
	if progressCount != 3 {
		t.Errorf("progress updates = %d, want 3", progressCount)
	}
	if doneCount != 1 {
		t.Errorf("done responses = %d, want 1", doneCount)
	}
	if imageCount != 1 {
		t.Errorf("image-bearing done responses = %d, want 1", imageCount)
	}

	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastImageParams.Prompt != "a cat" {
		t.Errorf("prompt = %q, want %q", mock.lastImageParams.Prompt, "a cat")
	}
	if mock.lastImageParams.Seed != 42 {
		t.Errorf("seed = %d, want 42", mock.lastImageParams.Seed)
	}
}

func TestHandlerImageError(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageErr:      errors.New("model load failed: corrupted weights"),
		progressSteps: 0,
	}
	s := newTestServer(ModeImage, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1})

	if len(resps) != 1 {
		t.Fatalf("responses = %d, want 1", len(resps))
	}
	if resps[0].Error == "" {
		t.Fatal("expected non-empty error in response")
	}
	if !strings.Contains(resps[0].Error, "corrupted weights") {
		t.Errorf("error = %q, want substring 'corrupted weights'", resps[0].Error)
	}
	if !resps[0].Done {
		t.Error("error response should have Done=true")
	}
}

func TestHandlerImageOOMError(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageErr:      errors.New("CUDA error: out of memory"),
	}
	s := newTestServer(ModeImage, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1})

	if len(resps) != 1 {
		t.Fatalf("responses = %d, want 1", len(resps))
	}
	if !strings.Contains(strings.ToLower(resps[0].Error), "out of memory") {
		t.Errorf("OOM error should contain 'out of memory', got %q", resps[0].Error)
	}
}

func TestHandlerVideoStreaming(t *testing.T) {
	frames := []sdcpp.Image{testImage(), testImage(), testImage()}
	mock := &mockSDContext{
		supportsVideo: true,
		videoResult:   frames,
		progressSteps: 2,
	}
	s := newTestServer(ModeVideo, mock)

	_, resps := doCompletion(t, s, DiffRequest{
		Prompt: "a dog running",
		Width:  832, Height: 480, Steps: 2, Seed: 7,
		VideoFrames: 3, FPS: 16, FlowShift: 3.0,
	})

	var progressCount, frameCount int
	var doneResp *DiffResponse
	for i := range resps {
		if resps[i].Done {
			doneResp = &resps[i]
		} else if resps[i].Step > 0 && resps[i].Frame == 0 && resps[i].Frames == 0 {
			progressCount++
		} else if resps[i].Frame > 0 || resps[i].Frames > 0 {
			if resps[i].Image != "" {
				frameCount++
			}
		}
	}
	if progressCount != 2 {
		t.Errorf("progress updates = %d, want 2", progressCount)
	}
	if frameCount != 3 {
		t.Errorf("frame responses with image = %d, want 3", frameCount)
	}
	if doneResp == nil {
		t.Fatal("missing done response")
	}
	if doneResp.Frames != 3 {
		t.Errorf("done.Frames = %d, want 3", doneResp.Frames)
	}

	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastVideoParams.VideoFrames != 3 {
		t.Errorf("VideoFrames = %d, want 3", mock.lastVideoParams.VideoFrames)
	}
	if mock.lastVideoParams.SampleParams.FlowShift != 3.0 {
		t.Errorf("FlowShift = %f, want 3.0", mock.lastVideoParams.SampleParams.FlowShift)
	}
}

func TestHandlerVideoError(t *testing.T) {
	mock := &mockSDContext{
		supportsVideo: true,
		videoErr:      errors.New("VAE decode failed"),
	}
	s := newTestServer(ModeVideo, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1, VideoFrames: 1})

	if len(resps) != 1 {
		t.Fatalf("responses = %d, want 1", len(resps))
	}
	if !strings.Contains(resps[0].Error, "VAE decode failed") {
		t.Errorf("error = %q, want substring 'VAE decode failed'", resps[0].Error)
	}
}

func TestHandlerVideoOOMError(t *testing.T) {
	mock := &mockSDContext{
		supportsVideo: true,
		videoErr:      errors.New("failed to allocate 14GB buffer"),
	}
	s := newTestServer(ModeVideo, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1, VideoFrames: 1})

	if !strings.Contains(strings.ToLower(resps[0].Error), "out of memory") {
		t.Errorf("OOM (failed to allocate) should be normalized to contain 'out of memory', got %q", resps[0].Error)
	}
}

func TestHandlerWarningPropagated(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeImage, mock)
	s.warning = "WAN video VAE does not support metal backend"

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1})

	var done *DiffResponse
	for i := range resps {
		if resps[i].Done {
			done = &resps[i]
		}
	}
	if done == nil {
		t.Fatal("missing done response")
	}
	if done.Warning != s.warning {
		t.Errorf("warning = %q, want %q", done.Warning, s.warning)
	}
}

func TestHandlerWarningOnErrorResponse(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageErr:      errors.New("boom"),
	}
	s := newTestServer(ModeImage, mock)
	s.warning = "VAE CPU fallback"

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1})

	if len(resps) != 1 {
		t.Fatalf("responses = %d, want 1", len(resps))
	}
	if resps[0].Warning != "VAE CPU fallback" {
		t.Errorf("error response warning = %q, want %q", resps[0].Warning, "VAE CPU fallback")
	}
}

func TestHandlerModeOverride(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		supportsVideo: true,
		imageResult:   []sdcpp.Image{testImage()},
		videoResult:   []sdcpp.Image{testImage()},
		progressSteps: 1,
	}
	s := newTestServer(ModeImage, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1, Mode: "video"})

	var done *DiffResponse
	for i := range resps {
		if resps[i].Done {
			done = &resps[i]
		}
	}
	if done == nil {
		t.Fatal("missing done response")
	}
	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastVideoParams.Prompt != "x" {
		t.Errorf("mode override to video did not call GenerateVideo; lastImage=%+v lastVideo=%+v", mock.lastImageParams, mock.lastVideoParams)
	}
}

func TestHandlerCancellation(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
		progressSteps: 0,
	}
	s := newTestServer(ModeImage, mock)

	body, _ := json.Marshal(DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1})
	ctx, cancel := context.WithCancel(context.Background())
	r := httptest.NewRequest(http.MethodPost, "/completion", bytes.NewReader(body)).WithContext(ctx)
	w := httptest.NewRecorder()

	cancel()
	s.completionHandler(w, r)

	if !mock.wasCancelled() {
		t.Error("expected CancelGeneration to be called after context cancellation")
	}
}

func TestHandlerSeedAutoGenerated(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeImage, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 0})

	if len(resps) == 0 {
		t.Fatal("no responses")
	}
	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastImageParams.Seed <= 0 {
		t.Errorf("seed should be auto-generated to a positive value, got %d", mock.lastImageParams.Seed)
	}
}

func TestHandlerBatchCountDefault(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeImage, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1, BatchCount: 0})

	if len(resps) == 0 {
		t.Fatal("no responses")
	}
	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastImageParams.BatchCount < 1 {
		t.Errorf("BatchCount should default to >=1, got %d", mock.lastImageParams.BatchCount)
	}
}

func TestHandlerNegativePromptForwarded(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeImage, mock)

	_, _ = doCompletion(t, s, DiffRequest{
		Prompt:         "a cat",
		NegativePrompt: "blurry, low quality",
		Width:          8, Height: 8, Steps: 1, Seed: 1,
	})

	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastImageParams.NegativePrompt != "blurry, low quality" {
		t.Errorf("NegativePrompt = %q, want %q", mock.lastImageParams.NegativePrompt, "blurry, low quality")
	}
}

func TestHandlerVideoEndImageForwarded(t *testing.T) {
	mock := &mockSDContext{
		supportsVideo: true,
		videoResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeVideo, mock)

	png1x1 := testPNGBytes()

	_, _ = doCompletion(t, s, DiffRequest{
		Prompt: "x",
		Width:  1, Height: 1, Steps: 1, Seed: 1,
		VideoFrames: 1, EndImage: png1x1,
	})

	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastVideoParams.EndImage == nil {
		t.Error("EndImage not forwarded to GenerateVideo")
	} else if mock.lastVideoParams.EndImage.Width != 1 {
		t.Errorf("EndImage width = %d, want 1", mock.lastVideoParams.EndImage.Width)
	}
}

func TestHandlerHealthEndpoint(t *testing.T) {
	mock := &mockSDContext{supportsImage: true}
	s := newTestServer(ModeImage, mock)

	r := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	s.healthHandler(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", w.Code)
	}
	var resp HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode health: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("health status = %q, want 'ok'", resp.Status)
	}
}

func TestHandlerCompletionMethodNotAllowed(t *testing.T) {
	mock := &mockSDContext{supportsImage: true}
	s := newTestServer(ModeImage, mock)

	r := httptest.NewRequest(http.MethodGet, "/completion", nil)
	w := httptest.NewRecorder()
	s.completionHandler(w, r)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", w.Code)
	}
}

func TestHandlerCompletionBadJSON(t *testing.T) {
	mock := &mockSDContext{supportsImage: true}
	s := newTestServer(ModeImage, mock)

	r := httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader("{bad json"))
	w := httptest.NewRecorder()
	s.completionHandler(w, r)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestHandlerVideoFramesSkippedOnError(t *testing.T) {
	mock := &mockSDContext{
		supportsVideo: true,
		videoErr:      errors.New("generate_video failed"),
	}
	s := newTestServer(ModeVideo, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 1, Seed: 1, VideoFrames: 5})

	for _, r := range resps {
		if r.Image != "" {
			t.Error("no frame images should be emitted on error")
		}
	}
}

func TestHandlerImageMultipleProgressOrder(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
		progressSteps: 5,
	}
	s := newTestServer(ModeImage, mock)

	_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 5, Seed: 1})

	var steps []int
	for _, r := range resps {
		if !r.Done && r.Step > 0 {
			steps = append(steps, r.Step)
		}
	}
	if len(steps) != 5 {
		t.Fatalf("progress count = %d, want 5", len(steps))
	}
	for i, st := range steps {
		if st != i+1 {
			t.Errorf("step[%d] = %d, want %d (ascending order)", i, st, i+1)
		}
	}
}

func TestHandlerCloseContext(t *testing.T) {
	mock := &mockSDContext{supportsImage: true}
	mock.Close()
	if !mock.closed {
		t.Error("Close should set closed flag")
	}
}

func TestHandlerImageParamsForwarded(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeImage, mock)

	_, _ = doCompletion(t, s, DiffRequest{
		Prompt:   "test prompt",
		Width:    512,
		Height:   512,
		Steps:    20,
		Seed:     99,
		CFGScale: 7.5,
	})

	mock.mu.Lock()
	defer mock.mu.Unlock()
	p := mock.lastImageParams
	if p.Width != 512 || p.Height != 512 {
		t.Errorf("dims = %dx%d, want 512x512", p.Width, p.Height)
	}
	if p.SampleParams.SampleSteps != 20 {
		t.Errorf("steps = %d, want 20", p.SampleParams.SampleSteps)
	}
	if p.SampleParams.CFGScale != 7.5 {
		t.Errorf("cfg = %f, want 7.5", p.SampleParams.CFGScale)
	}
}

func TestHandlerVideoParamsForwarded(t *testing.T) {
	mock := &mockSDContext{
		supportsVideo: true,
		videoResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeVideo, mock)

	_, _ = doCompletion(t, s, DiffRequest{
		Prompt:      "video prompt",
		Width:       832,
		Height:      480,
		Steps:       15,
		Seed:        42,
		CFGScale:    3.5,
		FlowShift:   3.0,
		VideoFrames: 33,
		FPS:         16,
	})

	mock.mu.Lock()
	defer mock.mu.Unlock()
	p := mock.lastVideoParams
	if p.Width != 832 || p.Height != 480 {
		t.Errorf("dims = %dx%d, want 832x480", p.Width, p.Height)
	}
	if p.VideoFrames != 33 {
		t.Errorf("VideoFrames = %d, want 33", p.VideoFrames)
	}
	if p.FPS != 16 {
		t.Errorf("FPS = %d, want 16", p.FPS)
	}
	if p.SampleParams.FlowShift != 3.0 {
		t.Errorf("FlowShift = %f, want 3.0", p.SampleParams.FlowShift)
	}
}

func TestHandlerImageInitImageForwarded(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
	}
	s := newTestServer(ModeImage, mock)

	png1x1 := testPNGBytes()

	_, _ = doCompletion(t, s, DiffRequest{
		Prompt: "x", Width: 1, Height: 1, Steps: 1, Seed: 1,
		Images: [][]byte{png1x1},
	})

	mock.mu.Lock()
	defer mock.mu.Unlock()
	if mock.lastImageParams.InitImage == nil {
		t.Fatal("InitImage not forwarded")
	}
	if mock.lastImageParams.InitImage.Width != 1 {
		t.Errorf("InitImage width = %d, want 1", mock.lastImageParams.InitImage.Width)
	}
}

func TestHandlerContextImplementsInterface(t *testing.T) {
	var _ sdContext = (*mockSDContext)(nil)
}

func TestHandlerNoDeadlockOnLongGeneration(t *testing.T) {
	mock := &mockSDContext{
		supportsImage: true,
		imageResult:   []sdcpp.Image{testImage()},
		progressSteps: 100,
	}
	s := newTestServer(ModeImage, mock)

	done := make(chan struct{})
	go func() {
		_, resps := doCompletion(t, s, DiffRequest{Prompt: "x", Width: 8, Height: 8, Steps: 100, Seed: 1})
		if len(resps) < 100 {
			t.Errorf("responses = %d, want >= 100", len(resps))
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("handler deadlocked or timed out")
	}
}
