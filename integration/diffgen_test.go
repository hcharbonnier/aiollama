//go:build integration

package integration

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// diffTestModel is set via OLLAMA_TEST_DIFF_MODEL. When set, the diffgen
// integration tests run against this model (image or video, auto-detected).
// When unset, the tests skip.
var diffTestModel = os.Getenv("OLLAMA_TEST_DIFF_MODEL")

func TestDiffgenImageGeneration(t *testing.T) {
	if diffTestModel == "" {
		t.Skip("OLLAMA_TEST_DIFF_MODEL not set; skipping diffgen image integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	pullOrSkip(ctx, t, client, diffTestModel)

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

	pullOrSkip(ctx, t, client, diffTestModel)

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

	pullOrSkip(ctx, t, client, diffTestModel)

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
