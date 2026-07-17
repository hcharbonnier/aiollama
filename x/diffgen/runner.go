//go:build sdcpp

package diffgen

import (
	"context"
	"encoding/base64"
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
	ctx       sdContext
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
			Guidance: sdcpp.GuidanceParams{
				TxtCfg: req.CFGScale,
			},
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
			Guidance: sdcpp.GuidanceParams{
				TxtCfg: req.CFGScale,
			},
			FlowShift: req.FlowShift,
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

	// warning accumulates s.warning (a structural, server-lifetime warning
	// such as the WAN VAE CPU fallback) plus any request-scoped notice below.
	// It is intentionally a local copy, not a mutation of s.warning, so it
	// doesn't leak across unrelated requests handled by this same runner.
	warning := s.warning

	// If the caller explicitly requested a WebM container and ffmpeg is
	// available on PATH, mux the frames into a single video blob instead of
	// streaming them individually. Falls back to the frame-stream protocol
	// below on any encoding failure (e.g. ffmpeg missing or erroring), in
	// which case the fallback is surfaced via the Warning field below rather
	// than only logged server-side.
	//
	// "webm-lossless" requests VP9 lossless (libvpx-vp9 -lossless 1); if the
	// resolved ffmpeg lacks libvpx-vp9, it transparently degrades to lossy
	// VP8 WebM (still a single container) and surfaces the downgrade via
	// Warning, before any PNG frame-stream fallback is considered.
	wantLossless := strings.EqualFold(req.OutputFormat, "webm-lossless")
	if (strings.EqualFold(req.OutputFormat, "webm") || wantLossless) && SupportsContainerEncoding() {
		lossless := wantLossless
		if lossless && !SupportsLosslessVP9() {
			slog.Warn("lossless VP9 requested but ffmpeg lacks libvpx-vp9; falling back to lossy VP8 webm", "model", s.modelName)
			fallbackNotice := "lossless VP9 (libvpx-vp9) unavailable in ffmpeg; returned lossy VP8 webm instead"
			if warning != "" {
				warning += "; " + fallbackNotice
			} else {
				warning = fallbackNotice
			}
			lossless = false
		}
		data, encErr := EncodeWebM(r.Context(), frames, req.FPS, lossless)
		if encErr == nil {
			resp := DiffResponse{
				Video: base64.StdEncoding.EncodeToString(data),
				Done:  true,
				// Frame equals Frames here (rather than 0) because no
				// incremental frame-decode progress is streamed on this
				// path; leaving Frame at its zero value would make
				// Completed/Total-based progress consumers (e.g. the CLI
				// step bar) appear to reset to 0 right as generation
				// finishes.
				Frame:   len(frames),
				Frames:  len(frames),
				Warning: warning,
			}
			enc.Encode(resp)
			w.Write([]byte("\n"))
			flusher.Flush()
			return
		}
		if r.Context().Err() != nil {
			return
		}
		slog.Warn("webm container encoding failed; falling back to frame stream", "model", s.modelName, "lossless", lossless, "error", encErr)
		fallbackNotice := "webm container encoding failed (" + encErr.Error() + "); returned individual frames instead"
		if warning != "" {
			warning += "; " + fallbackNotice
		} else {
			warning = fallbackNotice
		}
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

	resp := DiffResponse{Done: true, Frames: len(frames), Warning: warning}
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

	// SD.cpp distinguishes model_path (a full pipeline checkpoint, e.g. an
	// SD1.5 .ckpt) from diffusion_model_path (an isolated DiT/diffusion U-Net
	// weight, e.g. a FLUX.2 or WAN .gguf/.safetensors). Modern SD.cpp model
	// repos ship isolated diffusion weights, so prefer diffusion_model_path
	// when a "diffusion_model" component is present. Fall back to model_path
	// via the legacy "unet" component name for older SD1.x/SDXL checkpoints.
	if path, err := m.ComponentPath("diffusion_model"); err == nil {
		c.DiffusionModelPath = path
	} else if path, err := m.ComponentPath("unet"); err == nil {
		c.ModelPath = path
	} else {
		return nil, fmt.Errorf("no diffusion_model or unet component found")
	}

	// WAN 2.2 T2V/I2V A14B is a dual-stage model: a LowNoise diffusion model
	// (loaded above as the primary diffusion_model) plus a separate
	// HighNoise diffusion model. Without this, dual-stage generation produces
	// a black/incorrect video. Optional: absent for single-stage models.
	if path, err := m.ComponentPath("high_noise_diffusion_model"); err == nil {
		c.HighNoiseDiffusionModelPath = path
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
	if path, err := m.ComponentPath("llm"); err == nil {
		c.LLMPath = path
	}
	if path, err := m.ComponentPath("audio_vae"); err == nil {
		c.AudioVaePath = path
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
