//go:build sdcpp

package diffgen

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/diffgen/manifest"
	"github.com/ollama/ollama/x/sdcpp"
)

// diffGenMu serializes generation across a single runner subprocess because
// SD.cpp's generate_image/generate_video are synchronous and blocking.
var diffGenMu sync.Mutex

// Execute is the entry point for the diffgen runner subprocess
// (`ollama runner --diffgen-engine`).
func Execute(args []string) error {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: envconfig.LogLevel()})))

	sdcpp.SetLogCallback(func(level int32, text string) {
		msg := strings.TrimSpace(text)
		if msg == "" {
			return
		}
		switch level {
		case 0:
			slog.Debug("sdcpp", "msg", msg)
		case 1:
			slog.Info("sdcpp", "msg", msg)
		case 2:
			slog.Warn("sdcpp", "msg", msg)
		default:
			slog.Error("sdcpp", "msg", msg)
		}
	})

	fs := flag.NewFlagSet("diffgen-runner", flag.ExitOnError)
	modelName := fs.String("model", "", "path to model")
	port := fs.Int("port", 0, "port to listen on")
	backend := fs.String("backend", "", "SD.cpp backend: cuda/metal/vulkan/cpu")
	maxVRAM := fs.String("max-vram", "", "VRAM budget in GiB for layer offload")
	streamLayers := fs.Bool("stream-layers", false, "Enable SD.cpp layer streaming for VRAM offload")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelName == "" {
		return fmt.Errorf("--model is required")
	}
	if *port == 0 {
		return fmt.Errorf("--port is required")
	}

	srv, err := newRunnerServer(*modelName, *port, *backend, *maxVRAM, *streamLayers)
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.healthHandler)
	mux.HandleFunc("/completion", srv.completionHandler)

	httpServer := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", *port), Handler: mux}

	done := make(chan struct{})
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("shutting down diffgen runner")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			slog.Warn("graceful shutdown timed out", "error", err)
			httpServer.Close()
		}
		if srv.ctx != nil {
			srv.ctx.Close()
		}
		close(done)
	}()

	slog.Info("diffgen runner listening", "addr", httpServer.Addr)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}
	<-done
	return nil
}

type runnerServer struct {
	modelName string
	port      int
	ctx       *sdcpp.Context
	mode      ModelMode
	warning   string // non-fatal warning surfaced per-request (e.g. WAN VAE CPU fallback)
}

func newRunnerServer(modelName string, port int, backend, maxVRAMGiB string, streamLayers bool) (*runnerServer, error) {
	m, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest: %w", err)
	}

	configData, err := m.ReadModelIndex()
	if err != nil {
		return nil, fmt.Errorf("failed to read model_index.json: %w", err)
	}

	arch := DetectArchitecture(configData)
	modelType := DetectModelType(configData)
	slog.Info("detected diffgen model type", "type", modelType, "architecture", arch, "backend", backend)

	var warning string
	if WANVAEDeprecatedBackend(arch, backend) {
		warning = "WAN video VAE does not support " + backend + " backend; falling back to CPU VAE (significantly slower). CUDA or CPU is recommended for WAN video models."
		slog.Warn(warning, "architecture", arch, "backend", backend)
	}

	mode := ModeImage
	if modelType == "video" {
		mode = ModeVideo
	}

	ctx, err := createSDContext(m, backend, maxVRAMGiB, streamLayers)
	if err != nil {
		return nil, fmt.Errorf("failed to create SD.cpp context: %w", err)
	}

	// Validate capability matches detected mode.
	if mode == ModeVideo && !ctx.SupportsVideoGeneration() {
		return nil, fmt.Errorf("model detected as video but SD.cpp context does not support video generation")
	}
	if mode == ModeImage && !ctx.SupportsImageGeneration() {
		return nil, fmt.Errorf("model detected as image but SD.cpp context does not support image generation")
	}

	return &runnerServer{
		modelName: modelName,
		port:      port,
		ctx:       ctx,
		mode:      mode,
		warning:   warning,
	}, nil
}

func (s *runnerServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{Status: "ok"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *runnerServer) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req DiffRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	mode := s.mode
	if req.Mode == "image" {
		mode = ModeImage
	} else if req.Mode == "video" {
		mode = ModeVideo
	}

	switch mode {
	case ModeVideo:
		s.handleVideoCompletion(w, r, req)
	default:
		s.handleImageCompletion(w, r, req)
	}
}

// writeError sends an error response as ndjson. If the error is an OOM, the
// message is ensured to contain "out of memory" so the parent's
// llm.IsOutOfMemory can detect it and trigger eviction/retry. The warning
// field (if any) is included so callers see non-fatal warnings even on the
// error response.
func (s *runnerServer) writeError(enc *json.Encoder, w http.ResponseWriter, flusher http.Flusher, kind string, err error) {
	msg := err.Error()
	if llm.IsOutOfMemoryMessage(msg) {
		slog.Error("diffgen "+kind+" out of memory", "model", s.modelName, "error", err)
		if !strings.Contains(strings.ToLower(msg), "out of memory") {
			msg = "out of memory: " + msg
		}
	} else {
		slog.Error("diffgen "+kind+" failed", "model", s.modelName, "error", err)
	}
	resp := DiffResponse{Error: msg, Done: true, Warning: s.warning}
	enc.Encode(resp)
	w.Write([]byte("\n"))
	flusher.Flush()
}

func (s *runnerServer) handleImageCompletion(w http.ResponseWriter, r *http.Request, req DiffRequest) {
	diffGenMu.Lock()
	defer diffGenMu.Unlock()

	if req.Seed <= 0 {
		req.Seed = time.Now().UnixNano()
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	enc := json.NewEncoder(w)

	progress := func(step, steps int, seconds float32) {
		resp := DiffResponse{Step: step, Total: steps}
		enc.Encode(resp)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	go func() {
		<-r.Context().Done()
		s.ctx.CancelGeneration(sdcpp.CancelAll)
	}()

	var initImage *sdcpp.Image
	if len(req.Images) > 0 {
		if img, err := bytesToSDImage(req.Images[0]); err == nil {
			initImage = &img
		}
	}

	params := sdcpp.ImageGenParams{
		Prompt:          req.Prompt,
		NegativePrompt:  req.NegativePrompt,
		Width:           req.Width,
		Height:          req.Height,
		Seed:            req.Seed,
		BatchCount:      int32(max(1, req.BatchCount)),
		InitImage:       initImage,
		ControlStrength: req.ControlStrength,
		SampleParams: sdcpp.SampleParams{
			SampleSteps: int32(req.Steps),
			CFGScale:    req.CFGScale,
		},
	}

	images, err := s.ctx.GenerateImage(params, progress)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		s.writeError(enc, w, flusher, "image generation", err)
		return
	}

	b64, err := EncodeImageBase64(images[0])
	if err != nil {
		s.writeError(enc, w, flusher, "image encoding", err)
		return
	}

	resp := DiffResponse{Image: b64, Done: true, Warning: s.warning}
	enc.Encode(resp)
	w.Write([]byte("\n"))
	flusher.Flush()
}

func (s *runnerServer) handleVideoCompletion(w http.ResponseWriter, r *http.Request, req DiffRequest) {
	diffGenMu.Lock()
	defer diffGenMu.Unlock()

	if req.Seed <= 0 {
		req.Seed = time.Now().UnixNano()
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	enc := json.NewEncoder(w)

	progress := func(step, steps int, seconds float32) {
		resp := DiffResponse{Step: step, Total: steps}
		enc.Encode(resp)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	go func() {
		<-r.Context().Done()
		s.ctx.CancelGeneration(sdcpp.CancelAll)
	}()

	var initImage *sdcpp.Image
	if len(req.Images) > 0 {
		if img, err := bytesToSDImage(req.Images[0]); err == nil {
			initImage = &img
		}
	}
	var endImage *sdcpp.Image
	if len(req.EndImage) > 0 {
		if img, err := bytesToSDImage(req.EndImage); err == nil {
			endImage = &img
		}
	}

	params := sdcpp.VideoGenParams{
		Prompt:         req.Prompt,
		NegativePrompt: req.NegativePrompt,
		InitImage:      initImage,
		EndImage:       endImage,
		Width:          req.Width,
		Height:         req.Height,
		Seed:           req.Seed,
		VideoFrames:    int32(req.VideoFrames),
		FPS:            int32(req.FPS),
		SampleParams: sdcpp.SampleParams{
			SampleSteps: int32(req.Steps),
			CFGScale:    req.CFGScale,
			FlowShift:   req.FlowShift,
		},
	}

	frames, err := s.ctx.GenerateVideo(params, progress)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		s.writeError(enc, w, flusher, "video generation", err)
		return
	}

	for i, frame := range frames {
		b64, err := EncodeImageBase64(frame)
		if err != nil {
			continue
		}
		resp := DiffResponse{Frame: i, Frames: len(frames), Image: b64}
		enc.Encode(resp)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	resp := DiffResponse{Done: true, Frames: len(frames), Warning: s.warning}
	enc.Encode(resp)
	w.Write([]byte("\n"))
	flusher.Flush()
}

func createSDContext(m *manifest.ModelManifest, backend, maxVRAMGiB string, streamLayers bool) (*sdcpp.Context, error) {
	c := sdcpp.CtxParams{
		EnableMmap:   true,
		NThreads:     int32(numThreads()),
		Backend:      backend,
		MaxVRAM:      maxVRAMGiB,
		StreamLayers: streamLayers,
	}

	if path, err := m.ComponentPath("diffusion_model"); err == nil {
		c.ModelPath = path
	} else if path, err := m.ComponentPath("unet"); err == nil {
		c.ModelPath = path
	} else {
		return nil, fmt.Errorf("no diffusion_model or unet component found")
	}

	if path, err := m.ComponentPath("vae"); err == nil {
		c.VaePath = path
	}
	if path, err := m.ComponentPath("t5xxl"); err == nil {
		c.T5XXLPath = path
	}
	if path, err := m.ComponentPath("clip_l"); err == nil {
		c.ClipLPath = path
	}
	if path, err := m.ComponentPath("clip_g"); err == nil {
		c.ClipGPath = path
	}
	if path, err := m.ComponentPath("clip_vision"); err == nil {
		c.ClipVisionPath = path
	}
	if path, err := m.ComponentPath("taesd"); err == nil {
		c.TaesdPath = path
	}

	return sdcpp.NewContext(c)
}

func bytesToSDImage(data []byte) (sdcpp.Image, error) {
	img, err := DecodeImage(data)
	if err != nil {
		return sdcpp.Image{}, err
	}
	return ImageToSDImage(img)
}

func numThreads() int {
	return max(1, runtime.NumCPU())
}
