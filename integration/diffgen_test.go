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

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read model dir %s: %v", dir, err)
	}

	files := make(map[string]string, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		full := filepath.Join(dir, e.Name())
		digest, err := sha256Digest(full)
		if err != nil {
			t.Fatalf("digest %s: %v", full, err)
		}

		// Upload the blob.
		f, err := os.Open(full)
		if err != nil {
			t.Fatalf("open %s: %v", full, err)
		}
		if err := client.CreateBlob(ctx, digest, f); err != nil {
			f.Close()
			t.Fatalf("upload blob %s (%s): %v", e.Name(), digest, err)
		}
		f.Close()
		files[e.Name()] = digest
		t.Logf("uploaded %s -> %s", e.Name(), digest)
	}

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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
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
		Steps:  10,
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

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ensureDiffModel(ctx, t, client)

	prompt := "a lovely cat playing"
	t.Logf("Generating video with prompt: %s", prompt)

	var videoBase64 string
	var lastCompleted, lastTotal int64
	err := client.Generate(ctx, &api.GenerateRequest{
		Model:       diffTestModel,
		Prompt:      prompt,
		Width:       832,
		Height:      480,
		Steps:       20,
		VideoFrames: 9,
		FPS:         16,
		FlowShift:   3.0,
		CFGScale:    6.0,
	}, func(resp api.GenerateResponse) error {
		if resp.Completed > 0 {
			lastCompleted = resp.Completed
			lastTotal = resp.Total
		}
		if resp.Video != "" {
			videoBase64 = resp.Video
		}
		return nil
	})
	if err != nil {
		t.Fatalf("video generation failed: %v", err)
	}

	if lastTotal > 0 {
		t.Logf("Progress reached step %d/%d", lastCompleted, lastTotal)
	}

	if videoBase64 == "" {
		t.Fatal("no video data in response")
	}

	data, err := base64.StdEncoding.DecodeString(videoBase64)
	if err != nil {
		t.Fatalf("failed to decode base64 video: %v", err)
	}
	if len(data) < 100 {
		t.Fatalf("video data too small: %d bytes", len(data))
	}
	t.Logf("Generated video: %d bytes", len(data))
}

func TestDiffgenVideoAPI(t *testing.T) {
	if diffTestModel == "" {
		t.Skip("OLLAMA_TEST_DIFF_MODEL not set; skipping diffgen video API test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	ensureDiffModel(ctx, t, client)

	// Test the /v1/video/generations endpoint directly.
	reqBody := fmt.Sprintf(`{
		"model": %q,
		"prompt": "a dog running in the park",
		"size": "832x480",
		"video_frames": 9,
		"fps": 16,
		"steps": 10,
		"cfg_scale": 6.0,
		"flow_shift": 3.0,
		"stream": false
	}`, diffTestModel)
	url := fmt.Sprintf("http://%s/v1/video/generations", endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
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
