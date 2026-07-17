package videojobs

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

// ffmpegTranscoder implements Transcoder by piping PNG frames through an
// external ffmpeg process into a fragmented MP4 container. It is the default
// transcoder for the in-memory job store. ffmpeg is an optional runtime
// dependency looked up on PATH; if absent, Available() returns false and jobs
// fail fast with error code "ffmpeg_required".
type ffmpegTranscoder struct {
	once    sync.Once
	path    string
	pathErr error
}

// defaultTranscoder is the package-level singleton used by NewDefaultTranscoder.
var defaultTranscoder = &ffmpegTranscoder{}

// NewDefaultTranscoder returns the package-level singleton ffmpeg-based
// transcoder (looked up on PATH at first use).
func NewDefaultTranscoder() Transcoder {
	return defaultTranscoder
}

func (t *ffmpegTranscoder) lookup() (string, error) {
	t.once.Do(func() {
		t.path, t.pathErr = exec.LookPath("ffmpeg")
	})
	return t.path, t.pathErr
}

// Available reports whether ffmpeg is on PATH.
func (t *ffmpegTranscoder) Available() bool {
	_, err := t.lookup()
	return err == nil
}

// maxMP4OutputBytes bounds the encoded MP4 buffer. Mirrors the cap in
// x/diffgen/video.go (kept as a var here too, so tests can lower it to
// exercise the overflow path). Both definitions must stay in sync; this
// package does not import x/diffgen (which is behind the sdcpp build tag).
var maxMP4OutputBytes int64 = 512 * 1024 * 1024 // 512 MiB

// EncodeMP4 transcodes a sequence of PNG-encoded frames into a single
// fragmented MP4 (H.264, yuv420p) at fps. The frames are decoded to RGB and
// piped to ffmpeg as rawvideo. Fragmented MP4 (+frag_keyframe+empty_moov) is
// used because the MP4 muxer does not support non-seekable pipe output.
func (t *ffmpegTranscoder) EncodeMP4(ctx context.Context, framePNGs [][]byte, fps int) ([]byte, error) {
	if len(framePNGs) == 0 {
		return nil, errors.New("no frames to encode")
	}
	ffmpeg, err := t.lookup()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found on PATH: %w", err)
	}
	if fps <= 0 {
		fps = 16
	}

	// Decode the first frame to determine dimensions; validate the rest
	// match. PNG decode gives us the RGB width/height without needing the
	// image/* packages' Draw pipeline.
	w, h, err := pngFrameSize(framePNGs[0])
	if err != nil {
		return nil, fmt.Errorf("frame 0: %w", err)
	}
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("invalid frame dimensions %dx%d", w, h)
	}
	frameSize := w * h * 3

	args := []string{
		"-hide_banner", "-loglevel", "error", "-y",
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"-s", fmt.Sprintf("%dx%d", w, h),
		"-r", strconv.Itoa(fps),
		"-i", "-",
		"-an",
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		"-preset", "veryfast",
		"-crf", "23",
		"-movflags", "+frag_keyframe+empty_moov+default_base_moof",
		"-f", "mp4", "-",
	}
	cmd := exec.CommandContext(ctx, ffmpeg, args...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to open ffmpeg stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to open ffmpeg stdout: %w", err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start ffmpeg: %w", err)
	}

	// Decode each PNG to RGB and write to stdin on a goroutine so stdout
	// can be drained concurrently (matching the x/diffgen/video.go pattern).
	writeDone := make(chan error, 1)
	go func() {
		defer stdin.Close()
		for i, pngBytes := range framePNGs {
			rgb, err := pngToRGB(pngBytes, w, h, frameSize)
			if err != nil {
				writeDone <- fmt.Errorf("frame %d: %w", i, err)
				return
			}
			if _, err := stdin.Write(rgb); err != nil {
				writeDone <- err
				return
			}
		}
		writeDone <- nil
	}()

	var buf bytes.Buffer
	n, readErr := io.CopyN(&buf, stdout, maxMP4OutputBytes+1)
	if errors.Is(readErr, io.EOF) {
		readErr = nil
	}
	oversized := n > maxMP4OutputBytes
	if oversized {
		io.Copy(io.Discard, stdout)
	}

	runErr := cmd.Wait()
	writeErr := <-writeDone

	if runErr != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return nil, fmt.Errorf("ffmpeg mp4 encoding failed: %s", msg)
	}
	if writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
		return nil, fmt.Errorf("failed to stream frames to ffmpeg: %w", writeErr)
	}
	if readErr != nil {
		return nil, fmt.Errorf("failed to read ffmpeg output: %w", readErr)
	}
	if oversized {
		return nil, fmt.Errorf("ffmpeg output exceeded %d byte limit", maxMP4OutputBytes)
	}
	if buf.Len() == 0 {
		return nil, errors.New("ffmpeg produced no output")
	}
	return buf.Bytes(), nil
}

// pngFrameSize returns the dimensions of a PNG without fully decoding it.
func pngFrameSize(pngBytes []byte) (int, int, error) {
	cfg, err := png.DecodeConfig(bytes.NewReader(pngBytes))
	if err != nil {
		return 0, 0, fmt.Errorf("decode png config: %w", err)
	}
	return cfg.Width, cfg.Height, nil
}

// pngToRGB decodes a PNG to raw RGB24 bytes (w*h*3). The image is decoded to
// an NRGBA image and converted to RGB (NRGBA is the fastest general path for
// PNG; the alpha channel is dropped since the runner emits 3-channel frames).
func pngToRGB(pngBytes []byte, w, h, frameSize int) ([]byte, error) {
	img, err := png.Decode(bytes.NewReader(pngBytes))
	if err != nil {
		return nil, fmt.Errorf("decode png: %w", err)
	}
	b := img.Bounds()
	if b.Dx() != w || b.Dy() != h {
		return nil, fmt.Errorf("dimension mismatch %dx%d, want %dx%d", b.Dx(), b.Dy(), w, h)
	}
	rgb := make([]byte, frameSize)
	// Use a fast path for *image.NRGBA (common PNG decode result for
	// paletted/transparent PNGs) and *image.RGBA (the diffgen runner emits
	// RGBA via image.NewRGBA in x/diffgen/image.go). Both have a contiguous
	// Pix slice with stride == width*4, so RGB extraction is a tight loop.
	if nrgba, ok := img.(*image.NRGBA); ok && nrgba.Stride == w*4 {
		src := nrgba.Pix
		for i := 0; i < w*h; i++ {
			rgb[i*3+0] = src[i*4+0]
			rgb[i*3+1] = src[i*4+1]
			rgb[i*3+2] = src[i*4+2]
		}
		return rgb, nil
	}
	if rgba, ok := img.(*image.RGBA); ok && rgba.Stride == w*4 {
		src := rgba.Pix
		for i := 0; i < w*h; i++ {
			rgb[i*3+0] = src[i*4+0]
			rgb[i*3+1] = src[i*4+1]
			rgb[i*3+2] = src[i*4+2]
		}
		return rgb, nil
	}
	// Generic fallback via At (slower, but correct for any image type).
	idx := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rgb[idx+0] = byte(r >> 8)
			rgb[idx+1] = byte(g >> 8)
			rgb[idx+2] = byte(b >> 8)
			idx += 3
		}
	}
	return rgb, nil
}
