//go:build sdcpp

// Tests for WebM container encoding (EncodeWebM in video.go). These require
// ffmpeg on PATH; tests skip cleanly when it is unavailable, mirroring how
// the runner falls back to the PNG frame-stream protocol. Run with:
//
//	go test -tags=sdcpp ./x/diffgen/ -run TestEncodeWebM
//
// See docs/video-generation-implementation-plan.md §12.5.

package diffgen

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/x/sdcpp"
)

// solidFrame builds a small solid-color RGB frame.
func solidFrame(w, h int, r, g, b byte) sdcpp.Image {
	data := make([]byte, w*h*3)
	for i := 0; i < w*h; i++ {
		data[i*3+0] = r
		data[i*3+1] = g
		data[i*3+2] = b
	}
	return sdcpp.Image{Width: w, Height: h, Channel: 3, Data: data}
}

func requireFFmpeg(t *testing.T) {
	t.Helper()
	if !SupportsContainerEncoding() {
		t.Skip("ffmpeg not found on PATH; skipping WebM container encoding test")
	}
}

func TestEncodeWebMProducesValidContainer(t *testing.T) {
	requireFFmpeg(t)

	frames := []sdcpp.Image{
		solidFrame(64, 48, 255, 0, 0),
		solidFrame(64, 48, 0, 255, 0),
		solidFrame(64, 48, 0, 0, 255),
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	data, err := EncodeWebM(ctx, frames, 8)
	if err != nil {
		t.Fatalf("EncodeWebM failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("EncodeWebM returned no data")
	}
	// WebM/Matroska files start with the EBML header magic number.
	ebmlMagic := []byte{0x1A, 0x45, 0xDF, 0xA3}
	if len(data) < len(ebmlMagic) {
		t.Fatalf("output too short to contain EBML header: %d bytes", len(data))
	}
	for i, b := range ebmlMagic {
		if data[i] != b {
			t.Fatalf("output missing EBML magic header, got % x", data[:4])
		}
	}
}

func TestEncodeWebMNoFrames(t *testing.T) {
	_, err := EncodeWebM(context.Background(), nil, 16)
	if err == nil {
		t.Fatal("expected error for empty frame list")
	}
}

func TestEncodeWebMDimensionMismatch(t *testing.T) {
	requireFFmpeg(t)

	frames := []sdcpp.Image{
		solidFrame(64, 48, 255, 0, 0),
		solidFrame(32, 24, 0, 255, 0),
	}
	_, err := EncodeWebM(context.Background(), frames, 8)
	if err == nil {
		t.Fatal("expected error for mismatched frame dimensions")
	}
}

func TestEncodeWebMInvalidChannelCount(t *testing.T) {
	requireFFmpeg(t)

	frame := solidFrame(16, 16, 1, 2, 3)
	frame.Channel = 4 // not RGB
	_, err := EncodeWebM(context.Background(), []sdcpp.Image{frame}, 8)
	if err == nil {
		t.Fatal("expected error for non-RGB channel count")
	}
}

func TestEncodeWebMTruncatedData(t *testing.T) {
	requireFFmpeg(t)

	frame := sdcpp.Image{Width: 16, Height: 16, Channel: 3, Data: []byte{1, 2, 3}} // too short
	_, err := EncodeWebM(context.Background(), []sdcpp.Image{frame}, 8)
	if err == nil {
		t.Fatal("expected error for truncated frame data")
	}
}

func TestEncodeWebMDefaultFPS(t *testing.T) {
	requireFFmpeg(t)

	frames := []sdcpp.Image{solidFrame(16, 16, 10, 20, 30)}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// fps <= 0 should fall back to a sane default rather than erroring.
	data, err := EncodeWebM(ctx, frames, 0)
	if err != nil {
		t.Fatalf("EncodeWebM with fps=0 failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("EncodeWebM returned no data")
	}
}

func TestEncodeWebMContextCancellation(t *testing.T) {
	requireFFmpeg(t)

	frames := []sdcpp.Image{solidFrame(320, 240, 1, 2, 3)}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := EncodeWebM(ctx, frames, 8)
	if err == nil {
		t.Fatal("expected error when context is already cancelled")
	}
}

func TestSupportsContainerEncodingIsCached(t *testing.T) {
	first := SupportsContainerEncoding()
	second := SupportsContainerEncoding()
	if first != second {
		t.Errorf("SupportsContainerEncoding should be stable across calls: %v vs %v", first, second)
	}
}

// TestEncodeWebMOutputSizeCap verifies that output exceeding
// maxWebMOutputBytes is rejected (rather than being buffered without limit),
// and that ffmpeg is still allowed to exit cleanly (no hang/deadlock) when
// the cap is hit mid-stream. The cap is temporarily lowered so the test
// doesn't need to generate hundreds of megabytes of real video.
func TestEncodeWebMOutputSizeCap(t *testing.T) {
	requireFFmpeg(t)

	orig := maxWebMOutputBytes
	maxWebMOutputBytes = 1024 // 1 KiB: trivially exceeded by any real WebM output
	defer func() { maxWebMOutputBytes = orig }()

	// A reasonably large, low-compressibility frame set so the encoded
	// output exceeds the 1 KiB cap.
	frames := make([]sdcpp.Image, 10)
	for i := range frames {
		frames[i] = solidFrame(256, 256, byte(i*7), byte(i*13), byte(i*23))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := EncodeWebM(ctx, frames, 8)
	if err == nil {
		t.Fatal("expected an error when output exceeds the size cap")
	}
	if !strings.Contains(err.Error(), "exceeded") {
		t.Errorf("expected an 'exceeded' size-limit error, got: %v", err)
	}
}

// TestEncodeWebMUnderCapSucceeds is a regression guard ensuring the default
// cap does not reject small, legitimate output.
func TestEncodeWebMUnderCapSucceeds(t *testing.T) {
	requireFFmpeg(t)

	frames := []sdcpp.Image{solidFrame(32, 32, 1, 2, 3), solidFrame(32, 32, 4, 5, 6)}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	data, err := EncodeWebM(ctx, frames, 8)
	if err != nil {
		t.Fatalf("EncodeWebM failed for a small, legitimate input: %v", err)
	}
	if int64(len(data)) > maxWebMOutputBytes {
		t.Errorf("output (%d bytes) unexpectedly exceeds the default cap (%d bytes)", len(data), maxWebMOutputBytes)
	}
}
