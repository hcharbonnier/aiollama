//go:build sdcpp

// This file implements the diffgen Server (llm.LlamaServer) that wraps the
// SD.cpp runner subprocess. It is only compiled when the sdcpp build tag is
// set, which requires libstable-diffusion to be linked.

package diffgen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/diffgen/manifest"
)

// Server wraps an SD.cpp runner subprocess to implement llm.LlamaServer.
type Server struct {
	mu          sync.Mutex
	cmd         *exec.Cmd
	port        int
	modelName   string
	mode        string // "image" or "video"
	vramSize    uint64
	done        chan error
	client      *http.Client
	lastErr     string
	lastErrLock sync.Mutex
}

// NewServer prepares a new SD.cpp runner server. mode is "image" or "video".
// The subprocess is not started until Load() is called.
func NewServer(modelName, mode string) (llm.LlamaServer, error) {
	if err := CheckPlatformSupport(); err != nil {
		return nil, err
	}
	return &Server{
		modelName: modelName,
		mode:      mode,
		done:      make(chan error, 1),
		client:    &http.Client{Timeout: 30 * time.Minute},
	}, nil
}

func (s *Server) ModelPath() string {
	return s.modelName
}

// Load checks VRAM fit and spawns the subprocess.
func (s *Server) Load(ctx context.Context, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if m, err := manifest.LoadManifest(s.modelName); err == nil {
		s.vramSize = uint64(m.TotalComponentSize())
	} else {
		s.vramSize = 8 * 1024 * 1024 * 1024
	}

	backend := ResolveBackend(gpus)
	vramBudget := EstimateVRAMBudget(gpus, backend)
	// On CPU (no GPU budget), skip the pre-flight VRAM check: SD.cpp runs
	// entirely in system RAM and the OS swap handles oversubscription. The
	// check is only meaningful for GPU backends where OOM is a hard failure.
	if backend != "cpu" && s.vramSize > vramBudget {
		if requireFull {
			return nil, llm.ErrLoadRequiredFull
		}
		return nil, fmt.Errorf("model requires %s but only %s are available", format.HumanBytes2(s.vramSize), format.HumanBytes2(vramBudget))
	}

	maxVRAMGiB := FormatVRAMGiB(vramBudget)
	streamLayers := ShouldStreamLayers(s.vramSize, vramBudget)

	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152
	}
	s.port = port

	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	args := []string{"runner", "--diffgen-engine", "--model", s.modelName, "--port", strconv.Itoa(port)}
	if backend != "" {
		args = append(args, "--backend", backend)
	}
	if maxVRAMGiB != "" {
		args = append(args, "--max-vram", maxVRAMGiB)
	}
	if streamLayers {
		args = append(args, "--stream-layers")
	}
	cmd := exec.Command(exe, args...)
	cmd.Env = os.Environ()
	configureDiffgenSubprocessEnv(cmd, ml.LibraryPaths(gpus))
	s.cmd = cmd

	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			slog.Info("diffgen-runner", "msg", scanner.Text())
		}
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			slog.Warn("diffgen-runner", "msg", line)
			s.lastErrLock.Lock()
			s.lastErr = line
			s.lastErrLock.Unlock()
		}
	}()

	slog.Info("starting diffgen runner subprocess", "model", s.modelName, "port", s.port, "mode", s.mode)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start diffgen runner: %w", err)
	}
	go func() {
		s.done <- cmd.Wait()
	}()
	return nil, nil
}

func (s *Server) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}
	return nil
}

func diffgenLibraryPathEnv() string {
	switch runtime.GOOS {
	case "windows":
		return "PATH"
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	default:
		return "LD_LIBRARY_PATH"
	}
}

func configureDiffgenSubprocessEnv(cmd *exec.Cmd, libraryPaths []string) {
	if len(libraryPaths) == 0 {
		return
	}
	pathEnv := diffgenLibraryPathEnv()
	paths := append([]string{}, libraryPaths...)
	if existing, ok := os.LookupEnv(pathEnv); ok {
		paths = append(paths, filepath.SplitList(existing)...)
	}
	setSubprocessEnv(cmd, pathEnv, strings.Join(paths, string(filepath.ListSeparator)))
	ollamaPaths := append([]string{}, libraryPaths...)
	if existing, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); ok {
		ollamaPaths = append(ollamaPaths, filepath.SplitList(existing)...)
	}
	setSubprocessEnv(cmd, "OLLAMA_LIBRARY_PATH", strings.Join(ollamaPaths, string(filepath.ListSeparator)))
}

func setSubprocessEnv(cmd *exec.Cmd, key, value string) {
	for i := range cmd.Env {
		name, _, ok := strings.Cut(cmd.Env[i], "=")
		if ok && strings.EqualFold(name, key) {
			cmd.Env[i] = key + "=" + value
			return
		}
	}
	cmd.Env = append(cmd.Env, key+"="+value)
}

func (s *Server) getLastErr() string {
	s.lastErrLock.Lock()
	defer s.lastErrLock.Unlock()
	return s.lastErr
}

func (s *Server) WaitUntilRunning(ctx context.Context) error {
	timeout := time.After(envconfig.LoadTimeout())
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case err := <-s.done:
			if msg := s.getLastErr(); msg != "" {
				return fmt.Errorf("diffgen runner failed: %s (exit: %v)", msg, err)
			}
			return fmt.Errorf("diffgen runner exited unexpectedly: %w", err)
		case <-timeout:
			if msg := s.getLastErr(); msg != "" {
				return fmt.Errorf("timeout waiting for diffgen runner: %s", msg)
			}
			return errors.New("timeout waiting for diffgen runner to start")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				slog.Info("diffgen runner is ready", "port", s.port)
				return nil
			}
		}
	}
}

// Completion forwards a generation request to the subprocess /completion
// endpoint and streams ndjson responses back to the caller.
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	var images [][]byte
	for _, media := range req.Media {
		images = append(images, media.Data)
	}
	creq := DiffRequest{
		Prompt:         req.Prompt,
		Mode:           s.mode,
		Width:          req.Width,
		Height:         req.Height,
		Steps:          int(req.Steps),
		Seed:           seed,
		Images:         images,
		NegativePrompt: req.NegativePrompt,
		CFGScale:       req.CFGScale,
		Sampler:        req.Sampler,
		OutputFormat:   req.OutputFormat,
		VideoFrames:    int(req.VideoFrames),
		FPS:            int(req.FPS),
		FlowShift:      req.FlowShift,
		EndImage:       req.EndImage,
	}
	if req.Options != nil {
		creq.Options = &RequestOptions{
			NumPredict:  req.Options.NumPredict,
			Temperature: float64(req.Options.Temperature),
			TopP:        float64(req.Options.TopP),
			TopK:        req.Options.TopK,
			Stop:        req.Options.Stop,
		}
	}
	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("http://127.0.0.1:%d/completion", s.port), bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s", strings.TrimSpace(string(b)))
	}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	for scanner.Scan() {
		var raw struct {
			Image   string `json:"image,omitempty"`
			Video   string `json:"video,omitempty"`
			Content string `json:"content,omitempty"`
			Done    bool   `json:"done"`
			Step    int    `json:"step,omitempty"`
			Total   int    `json:"total,omitempty"`
			Frame   int    `json:"frame,omitempty"`
			Frames  int    `json:"frames,omitempty"`
			Error   string `json:"error,omitempty"`
			Warning string `json:"warning,omitempty"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			continue
		}
		if raw.Warning != "" {
			slog.Warn("diffgen runner warning", "model", s.modelName, "warning", raw.Warning)
		}
		// Map video frame progress onto Step/TotalSteps so the CLI progress
		// bar works for video too. For image mode, Step/Total come from the
		// diffusion step callbacks directly.
		step := raw.Step
		total := raw.Total
		if raw.Frames > 0 {
			step = raw.Frame
			total = raw.Frames
		}
		cresp := llm.CompletionResponse{
			Content:    raw.Content,
			Done:       raw.Done,
			Step:       step,
			TotalSteps: total,
			Image:      raw.Image,
			Video:      raw.Video,
		}
		fn(cresp)
		if raw.Error != "" {
			return fmt.Errorf("%s", raw.Error)
		}
		if cresp.Done {
			return nil
		}
	}
	if s.HasExited() {
		if msg := s.getLastErr(); msg != "" {
			return fmt.Errorf("diffgen runner closed response: %s", msg)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return errors.New("diffgen runner closed response before completion")
}

func (s *Server) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return errors.New("diffgen runner does not support native chat")
}

func (s *Server) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return "", errors.New("diffgen runner does not support chat templates")
}

func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		slog.Info("stopping diffgen runner subprocess", "pid", s.cmd.Process.Pid)
		s.cmd.Process.Signal(os.Interrupt)
		select {
		case <-s.done:
		case <-time.After(5 * time.Second):
			s.cmd.Process.Kill()
		}
		s.cmd = nil
	}
	return nil
}

func (s *Server) MemorySize() (total, vram uint64) {
	return s.vramSize, s.vramSize
}

func (s *Server) VRAMByGPU(id ml.DeviceID) uint64 {
	return s.vramSize
}

func (s *Server) ContextLength() int { return 0 }

func (s *Server) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("embeddings not supported for diffgen models")
}

func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("tokenization not supported for diffgen models")
}

func (s *Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("detokenization not supported for diffgen models")
}

func (s *Server) Pid() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return -1
}

func (s *Server) GetPort() int { return s.port }

func (s *Server) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo { return nil }

func (s *Server) HasExited() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

var _ llm.LlamaServer = (*Server)(nil)
