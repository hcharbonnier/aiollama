//go:build integration

package integration

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// diffTestModel is set via OLLAMA_TEST_DIFF_MODEL. When set, the diffgen
// integration tests run against this model (image or video, auto-detected).
// When unset, the tests skip.
var diffTestModel = os.Getenv("OLLAMA_TEST_DIFF_MODEL")

// diffTestModelDir is set via OLLAMA_TEST_DIFF_MODEL_DIR. When set alongside
// OLLAMA_TEST_DIFF_MODEL, the tests import the model from a local directory of
// SD.cpp component files (model_index.json + weights) before running, instead
// of pulling from the registry. This is the path used for E2E testing with real
// (or dummy) SD.cpp model weights outside the Ollama registry.
var diffTestModelDir = os.Getenv("OLLAMA_TEST_DIFF_MODEL_DIR")

// diffTestVideoParams holds optional overrides for the video E2E test params
// (width/height/steps/frames/fps/cfg_scale/flow_shift/timeout/size). Each is
// controlled by an OLLAMA_TEST_DIFF_VIDEO_* env var and falls back to a
// CPU-friendly default when unset. On a GPU runner the defaults can be raised
// (e.g. 832x480, 20 steps, 33 frames) for a fuller test; on CPU the defaults
// keep the test tractable (1 frame, 4 steps — enough to exercise the full
// generate_video → VAE decode → frame encode pipeline).
var diffTestVideoParams = struct {
	Width, Height, Steps, VideoFrames, FPS int
	CFGScale, FlowShift                    float32
	Size                                   string
	Timeout                                time.Duration
}{
	Width:       envInt("OLLAMA_TEST_DIFF_VIDEO_WIDTH", 832),
	Height:      envInt("OLLAMA_TEST_DIFF_VIDEO_HEIGHT", 480),
	Steps:       envInt("OLLAMA_TEST_DIFF_VIDEO_STEPS", 4),
	VideoFrames: envInt("OLLAMA_TEST_DIFF_VIDEO_FRAMES", 1),
	FPS:         envInt("OLLAMA_TEST_DIFF_VIDEO_FPS", 16),
	CFGScale:    envFloat32("OLLAMA_TEST_DIFF_VIDEO_CFG_SCALE", 6.0),
	FlowShift:   envFloat32("OLLAMA_TEST_DIFF_VIDEO_FLOW_SHIFT", 3.0),
	Size:        os.Getenv("OLLAMA_TEST_DIFF_VIDEO_SIZE"),
	Timeout:     envDuration("OLLAMA_TEST_DIFF_VIDEO_TIMEOUT", 90*time.Minute),
}

// envInt parses an integer env var, returning fallback when unset or invalid.
func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

// envFloat32 parses a float32 env var, returning fallback when unset or invalid.
func envFloat32(key string, fallback float32) float32 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			return float32(f)
		}
	}
	return fallback
}

// envDuration parses a duration env var, returning fallback when unset/invalid.
func envDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}

// sha256Digest computes the sha256:... digest of a file's contents, mirroring
// parser.digestForFile so the server's blob store accepts the upload.
func sha256Digest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil)), nil
}

// importDiffModelFromDir uploads the component files in dir as blobs and
// creates a model named modelName via the /api/create endpoint. The directory
// must contain a model_index.json plus the component files it references. The
// files map uses forward-slash relative paths (matching the CLI's
// createRequestFileNames convention for a single shared root). Returns nil on
// success or if the model already exists.
func importDiffModelFromDir(ctx context.Context, t *testing.T, client *api.Client, dir, modelName string) {
	t.Helper()

	// If the model already exists, skip the import.
	if _, err := client.Show(ctx, &api.ShowRequest{Name: modelName}); err == nil {
		t.Logf("model %s already exists; skipping import", modelName)
		return
	}

	files := make(map[string]string)
	// Walk recursively so repos with subdirectory layouts (e.g. WAN 2.2's
	// LowNoise/, HighNoise/, VAE/) can be imported from their original
	// directory structure without manual flattening. model_index.json and
	// component files at any depth are included; the path key uses
	// forward slashes to match the CLI's createRequestFileNames convention.
	var walk func(d string)
	walk = func(d string) {
		sub, err := os.ReadDir(d)
		if err != nil {
			t.Fatalf("read subdir %s: %v", d, err)
		}
		for _, e := range sub {
			full := filepath.Join(d, e.Name())
			if e.IsDir() {
				walk(full)
				continue
			}
			rel, err := filepath.Rel(dir, full)
			if err != nil {
				t.Fatalf("rel path %s: %v", full, err)
			}
			rel = filepath.ToSlash(rel)
			digest, err := sha256Digest(full)
			if err != nil {
				t.Fatalf("digest %s: %v", full, err)
			}
			f, err := os.Open(full)
			if err != nil {
				t.Fatalf("open %s: %v", full, err)
			}
			if err := client.CreateBlob(ctx, digest, f); err != nil {
				f.Close()
				t.Fatalf("upload blob %s (%s): %v", rel, digest, err)
			}
			f.Close()
			files[rel] = digest
			t.Logf("uploaded %s -> %s", rel, digest)
		}
	}
	walk(dir)

	stream := false
	if err := client.Create(ctx, &api.CreateRequest{
		Model: modelName,
		Files: files,
		Stream: &stream,
	}, func(api.ProgressResponse) error { return nil }); err != nil {
		t.Fatalf("create model %s from dir %s: %v", modelName, dir, err)
	}
}

// ensureDiffModel makes diffTestModel available on the server, either by
// importing it from diffTestModelDir (local SD.cpp component files) or by
// pulling from the registry. It skips the test if neither path is configured.
func ensureDiffModel(ctx context.Context, t *testing.T, client *api.Client) {
	if diffTestModelDir != "" {
		importDiffModelFromDir(ctx, t, client, diffTestModelDir, diffTestModel)
	} else {
		pullOrSkip(ctx, t, client, diffTestModel)
	}
}

func TestDiffgenImageGeneration(t *testing.T) {
	if diffTestModel == "" {
		t.Skip("OLLAMA_TEST_DIFF_MODEL not set; skipping diffgen image integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ensureDiffModel(ctx, t, client)

	prompt := "a lovely cat on the moon, high quality"
	t.Logf("Generating image with prompt: %s", prompt)

	var imageBase64 string
	err := client.Generate(ctx, &api.GenerateRequest{
		Model:  diffTestModel,
		Prompt: prompt,
		Width:  512,
		Height: 512,
		Steps:  4,
	}, func(resp api.GenerateResponse) error {
		if resp.Image != "" {
			imageBase64 = resp.Image
		}
		return nil
	})
	if err != nil {
		t.Fatalf("image generation failed: %v", err)
	}

	if imageBase64 == "" {
		t.Fatal("no image data in response")
	}

	data, err := base64.StdEncoding.DecodeString(imageBase64)
	if err != nil {
		t.Fatalf("failed to decode base64 image: %v", err)
	}
	if len(data) < 100 {
		t.Fatalf("image data too small: %d bytes", len(data))
	}
	t.Logf("Generated image: %d bytes", len(data))
}

func TestDiffgenVideoGeneration(t *testing.T) {
	if diffTestModel == "" {
		t.Skip("OLLAMA_TEST_DIFF_MODEL not set; skipping diffgen video integration test")
	}

	p := diffTestVideoParams
	ctx, cancel := context.WithTimeout(context.Background(), p.Timeout)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ensureDiffModel(ctx, t, client)

	prompt := "a lovely cat playing"
	t.Logf("Generating video with prompt: %s (%dx%d, %d frames, %d steps)",
		prompt, p.Width, p.Height, p.VideoFrames, p.Steps)

	var videoBase64 string
	var frameImages []string
	var lastCompleted, lastTotal int64
	err := client.Generate(ctx, &api.GenerateRequest{
		Model:       diffTestModel,
		Prompt:      prompt,
		Width:       int32(p.Width),
		Height:      int32(p.Height),
		Steps:       int32(p.Steps),
		VideoFrames: int32(p.VideoFrames),
		FPS:         int32(p.FPS),
		FlowShift:   p.FlowShift,
		CFGScale:    p.CFGScale,
	}, func(resp api.GenerateResponse) error {
		if resp.Completed > 0 {
			lastCompleted = resp.Completed
			lastTotal = resp.Total
		}
		if resp.Video != "" {
			videoBase64 = resp.Video
		}
		if resp.Image != "" {
			frameImages = append(frameImages, resp.Image)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("video generation failed: %v", err)
	}

	if lastTotal > 0 {
		t.Logf("Progress reached step %d/%d", lastCompleted, lastTotal)
	}

	// The runner returns either a single video container (when
	// output_format is set) or individual PNG frames (frame stream).
	// Either is acceptable.
	if videoBase64 == "" && len(frameImages) == 0 {
		t.Fatal("no video data or frames in response")
	}

	if videoBase64 != "" {
		data, err := base64.StdEncoding.DecodeString(videoBase64)
		if err != nil {
			t.Fatalf("failed to decode base64 video: %v", err)
		}
		if len(data) < 100 {
			t.Fatalf("video data too small: %d bytes", len(data))
		}
		t.Logf("Generated video: %d bytes", len(data))
	} else {
		t.Logf("Generated %d frame images (frame stream protocol)", len(frameImages))
		if len(frameImages) > 0 {
			data, err := base64.StdEncoding.DecodeString(frameImages[0])
			if err != nil {
				t.Fatalf("failed to decode base64 frame image: %v", err)
			}
			if len(data) < 100 {
				t.Fatalf("frame image too small: %d bytes", len(data))
			}
			t.Logf("First frame: %d bytes", len(data))
		}
	}
}

func TestDiffgenVideoAPI(t *testing.T) {
	if diffTestModel == "" {
		t.Skip("OLLAMA_TEST_DIFF_MODEL not set; skipping diffgen video API test")
	}

	p := diffTestVideoParams
	ctx, cancel := context.WithTimeout(context.Background(), p.Timeout)
	defer cancel()

	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ensureDiffModel(ctx, t, client)

	// Test the /v1/video/generations endpoint directly.
	size := p.Size
	if size == "" {
		size = fmt.Sprintf("%dx%d", p.Width, p.Height)
	}
	reqBody := fmt.Sprintf(`{
		"model": %q,
		"prompt": "a dog running in the park",
		"size": %q,
		"video_frames": %d,
		"fps": %d,
		"steps": %d,
		"cfg_scale": %f,
		"flow_shift": %f,
		"stream": false
	}`, diffTestModel, size, p.VideoFrames, p.FPS, p.Steps, p.CFGScale, p.FlowShift)
	url := fmt.Sprintf("http://%s/v1/video/generations", endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	t.Logf("POST %s (size=%s, %d frames, %d steps)", url, size, p.VideoFrames, p.Steps)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("video generations request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}
	t.Logf("Video API returned status %d", resp.StatusCode)
}

// TestDiffgenImageGenerationProgress verifies that the streaming response
// includes step/total progress events before the final image. This exercises
// the ndjson progress streaming contract that the diffgen runner emits.
func TestDiffgenImageGenerationProgress(t *testing.T) {
	if diffTestModel == "" {
		t.Skip("OLLAMA_TEST_DIFF_MODEL not set; skipping diffgen image progress test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ensureDiffModel(ctx, t, client)

	var progressEvents int
	var finalImage string
	err := client.Generate(ctx, &api.GenerateRequest{
		Model:  diffTestModel,
		Prompt: "a small house in a forest, high quality",
		Width:  512,
		Height: 512,
		Steps:  4,
	}, func(resp api.GenerateResponse) error {
		// Progress events carry step/total without a final image.
		if resp.Image == "" && (resp.Total > 0 || resp.Completed > 0) {
			progressEvents++
		}
		if resp.Image != "" {
			finalImage = resp.Image
		}
		return nil
	})
	if err != nil {
		t.Fatalf("image generation failed: %v", err)
	}

	if progressEvents == 0 {
		t.Error("expected at least one progress event before the final image, got none")
	}
	if finalImage == "" {
		t.Fatal("no image data in response")
	}
	t.Logf("Generated image (%d bytes) after %d progress events", len(finalImage), progressEvents)
}

// TestDiffgenImportFromDirectory exercises the SD.cpp model import path
// (convertFromSDCpp) end-to-end: it creates a model from a local directory of
// component files and verifies the model appears in /api/tags with the correct
// capabilities. This does NOT require a real model or GPU — it only tests the
// import/manifest path. Set OLLAMA_TEST_DIFF_IMPORT_DIR to a directory with a
// model_index.json + dummy component files. The model is created under the name
// in OLLAMA_TEST_DIFF_MODEL (or a temp name if unset).
func TestDiffgenImportFromDirectory(t *testing.T) {
	importDir := os.Getenv("OLLAMA_TEST_DIFF_IMPORT_DIR")
	if importDir == "" {
		t.Skip("OLLAMA_TEST_DIFF_IMPORT_DIR not set; skipping diffgen import test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	modelName := diffTestModel
	if modelName == "" {
		modelName = "test-diffgen-import"
	}

	importDiffModelFromDir(ctx, t, client, importDir, modelName)

	// Verify the model is listed.
	list, err := client.List(ctx)
	if err != nil {
		t.Fatalf("list models: %v", err)
	}
	var found bool
	for _, m := range list.Models {
		if m.Name == modelName || strings.HasPrefix(m.Name, modelName+":") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("imported model %q not found in model list", modelName)
	}

	// Verify the model can be shown (manifest is valid).
	show, err := client.Show(ctx, &api.ShowRequest{Name: modelName})
	if err != nil {
		t.Fatalf("show model %s: %v", modelName, err)
	}
	t.Logf("imported model %s: capabilities=%v", modelName, show.Details.Families)
}
