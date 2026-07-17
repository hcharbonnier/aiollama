package videojobs

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"os/exec"
	"testing"
	"time"
)

func requireFFmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not found on PATH; skipping transcoder integration test")
	}
}

// pngFrame builds a small solid-color PNG-encoded frame.
func pngFrame(w, h int, c color.RGBA) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

func TestFFmpegTranscoderAvailable(t *testing.T) {
	tc := &ffmpegTranscoder{}
	if !tc.Available() {
		t.Skip("ffmpeg not on PATH")
	}
}

func TestFFmpegTranscoderEncodeMP4(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	frames := [][]byte{
		pngFrame(64, 48, color.RGBA{255, 0, 0, 255}),
		pngFrame(64, 48, color.RGBA{0, 255, 0, 255}),
		pngFrame(64, 48, color.RGBA{0, 0, 255, 255}),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	data, err := tc.EncodeMP4(ctx, frames, 8)
	if err != nil {
		t.Fatalf("EncodeMP4 failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("EncodeMP4 returned no data")
	}
	// Validate the MP4 ftyp box header.
	ftyp := []byte("ftyp")
	if len(data) < 8 || string(data[4:8]) != string(ftyp) {
		t.Fatalf("output missing ftyp box header, got % x", data[:min(8, len(data))])
	}
}

func TestFFmpegTranscoderNoFrames(t *testing.T) {
	tc := &ffmpegTranscoder{}
	_, err := tc.EncodeMP4(context.Background(), nil, 16)
	if err == nil {
		t.Fatal("expected error for empty frame list")
	}
}

func TestFFmpegTranscoderDimensionMismatch(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	frames := [][]byte{
		pngFrame(64, 48, color.RGBA{1, 2, 3, 255}),
		pngFrame(32, 24, color.RGBA{4, 5, 6, 255}),
	}
	_, err := tc.EncodeMP4(context.Background(), frames, 8)
	if err == nil {
		t.Fatal("expected error for mismatched frame dimensions")
	}
}

func TestFFmpegTranscoderContextCancellation(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	frames := [][]byte{pngFrame(320, 240, color.RGBA{1, 2, 3, 255})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := tc.EncodeMP4(ctx, frames, 8)
	if err == nil {
		t.Fatal("expected error when context is already cancelled")
	}
}

func TestFFmpegTranscoderEndToEndJob(t *testing.T) {
	requireFFmpeg(t)
	store := NewJobStoreWithConcurrency(NewDefaultTranscoder(), MaxConcurrentJobs)
	defer store.Close()

	// Build a GenerateFunc that emits 3 real PNG frames + progress.
	gen := func(ctx context.Context, params CreateParams, fn func(framePNG []byte, step, total int)) error {
		fn(nil, 5, 10)
		fn(nil, 10, 10)
		for i := 0; i < 3; i++ {
			fn(pngFrame(32, 32, color.RGBA{byte(i * 80), byte(i * 40), byte(i * 20), 255}), 0, 0)
		}
		return nil
	}

	j, err := store.Create(CreateParams{
		Model: "wan2.1-t2v", Prompt: "test", Seconds: "4", Size: "720x1280",
		Generate: gen,
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if j.Status() == "completed" || j.Status() == "failed" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	if j.Status() != "completed" {
		v := j.ToVideo()
		t.Fatalf("job did not complete: status=%q err=%+v", j.Status(), v.Error)
	}

	content, ct := j.Content()
	if len(content) == 0 {
		t.Fatal("expected non-empty MP4 content")
	}
	if ct != "video/mp4" {
		t.Errorf("content type = %q, want video/mp4", ct)
	}
	if len(content) < 8 || string(content[4:8]) != "ftyp" {
		t.Fatalf("content missing ftyp box header: % x", content[:min(8, len(content))])
	}
}
