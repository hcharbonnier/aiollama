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

// TestFFmpegTranscoderDecodeFrames verifies that DecodeFrames extracts PNG
// frames from an MP4 produced by EncodeMP4 (round-trip). Requires ffmpeg.
func TestFFmpegTranscoderDecodeFrames(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	// Encode 3 frames, then decode them back.
	frames := [][]byte{
		pngFrame(64, 48, color.RGBA{255, 0, 0, 255}),
		pngFrame(64, 48, color.RGBA{0, 255, 0, 255}),
		pngFrame(64, 48, color.RGBA{0, 0, 255, 255}),
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	mp4, err := tc.EncodeMP4(ctx, frames, 8)
	if err != nil {
		t.Fatalf("EncodeMP4 failed: %v", err)
	}

	decoded, _, err := tc.DecodeFrames(ctx, mp4, 0)
	if err != nil {
		t.Fatalf("DecodeFrames failed: %v", err)
	}
	if len(decoded) < 3 {
		t.Fatalf("expected at least 3 decoded frames, got %d", len(decoded))
	}
	// Each decoded frame should be a valid PNG (re-decodable).
	for i, f := range decoded {
		if _, err := png.Decode(bytes.NewReader(f)); err != nil {
			t.Errorf("decoded frame %d is not a valid PNG: %v", i, err)
		}
	}
}

// TestFFmpegTranscoderDecodeFramesEmpty verifies that decoding an empty input
// returns an error.
func TestFFmpegTranscoderDecodeFramesEmpty(t *testing.T) {
	tc := &ffmpegTranscoder{}
	_, _, err := tc.DecodeFrames(context.Background(), nil, 0)
	if err == nil {
		t.Fatal("expected error for empty video input")
	}
}

// TestFFmpegTranscoderConcatMP4 verifies that ConcatMP4 concatenates two MP4s
// into a single valid MP4. Requires ffmpeg.
func TestFFmpegTranscoderConcatMP4(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	frames := [][]byte{
		pngFrame(64, 48, color.RGBA{255, 0, 0, 255}),
		pngFrame(64, 48, color.RGBA{0, 255, 0, 255}),
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	first, err := tc.EncodeMP4(ctx, frames[:1], 8)
	if err != nil {
		t.Fatalf("encode first: %v", err)
	}
	second, err := tc.EncodeMP4(ctx, frames[1:], 8)
	if err != nil {
		t.Fatalf("encode second: %v", err)
	}

	concat, err := tc.ConcatMP4(ctx, first, second, 8)
	if err != nil {
		t.Fatalf("ConcatMP4 failed: %v", err)
	}
	if len(concat) < 8 || string(concat[4:8]) != "ftyp" {
		t.Fatalf("concat output missing ftyp box header: % x", concat[:min(8, len(concat))])
	}

	// The concatenated MP4 should decode to >= 2 frames.
	decoded, _, err := tc.DecodeFrames(ctx, concat, 0)
	if err != nil {
		t.Fatalf("decode concat: %v", err)
	}
	if len(decoded) < 2 {
		t.Errorf("expected >= 2 frames from concatenated MP4, got %d", len(decoded))
	}
}

// TestFFmpegTranscoderConcatMP4EmptyInputs verifies that concat with an empty
// input returns the other input unchanged (no ffmpeg call needed).
func TestFFmpegTranscoderConcatMP4EmptyInputs(t *testing.T) {
	tc := &ffmpegTranscoder{}
	solo := []byte{0, 0, 0, 0x18, 'f', 't', 'y', 'p'}

	got, err := tc.ConcatMP4(context.Background(), nil, solo, 16)
	if err != nil || !bytes.Equal(got, solo) {
		t.Errorf("concat(nil, solo) = %v, %v; want solo, nil", got, err)
	}
	got, err = tc.ConcatMP4(context.Background(), solo, nil, 16)
	if err != nil || !bytes.Equal(got, solo) {
		t.Errorf("concat(solo, nil) = %v, %v; want solo, nil", got, err)
	}
}

func TestFFmpegTranscoderProbeDurationSeconds(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	// 16 frames at 8 fps = 2 seconds.
	frames := make([][]byte, 16)
	for i := range frames {
		frames[i] = pngFrame(64, 48, color.RGBA{255, 0, 0, 255})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	mp4, err := tc.EncodeMP4(ctx, frames, 8)
	if err != nil {
		t.Fatalf("EncodeMP4: %v", err)
	}

	secs, err := tc.ProbeDurationSeconds(ctx, mp4)
	if err != nil {
		t.Fatalf("ProbeDurationSeconds: %v", err)
	}
	if secs != 2 {
		t.Errorf("duration = %d, want 2", secs)
	}

	if _, err := tc.ProbeDurationSeconds(ctx, nil); err == nil {
		t.Error("expected error for empty input")
	}
}

func TestFFmpegTranscoderSpritesheet(t *testing.T) {
	requireFFmpeg(t)
	tc := &ffmpegTranscoder{}

	frames := make([][]byte, 16)
	for i := range frames {
		frames[i] = pngFrame(64, 48, color.RGBA{0, 255, 0, 255})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	mp4, err := tc.EncodeMP4(ctx, frames, 8)
	if err != nil {
		t.Fatalf("EncodeMP4: %v", err)
	}

	sheet, err := tc.Spritesheet(ctx, mp4)
	if err != nil {
		t.Fatalf("Spritesheet: %v", err)
	}
	img, err := png.Decode(bytes.NewReader(sheet))
	if err != nil {
		t.Fatalf("spritesheet is not a PNG: %v", err)
	}
	// Tiled sheet should be larger than a single 64x48 frame.
	if img.Bounds().Dx() < 64 || img.Bounds().Dy() < 48 {
		t.Errorf("spritesheet too small: %v", img.Bounds())
	}

	if _, err := tc.Spritesheet(ctx, nil); err == nil {
		t.Error("expected error for empty input")
	}
}
