//go:build sdcpp

package diffgen

// Video container encoding. WebM is produced by piping raw RGB frames through
// an external `ffmpeg` process (rawvideo -> libvpx/libvpx-vp9 -> webm). This
// avoids vendoring a VP8/VP9 encoder or linking cgo ffmpeg bindings: ffmpeg is
// an optional *runtime* dependency looked up on PATH, not a build-time Go
// module dependency, so binaries that ship without ffmpeg still build and run
// — they simply fall back to the PNG frame-stream protocol (see
// handleVideoCompletion in runner.go). This addresses the "binary bloat /
// cross-compilation pain" risk called out in the implementation plan (§6)
// for container encoding while still delivering a real single-file video
// container per §12.5.
//
// Two codecs are supported, selected via output_format:
//   - "webm": VP8 (libvpx). Broad availability, compact lossy output. The
//     default single-file container format.
//   - "webm-lossless": VP9 lossless (libvpx-vp9 -lossless 1). Pixel-perfect
//     preservation of SD.cpp's VAE decoder output at the cost of much larger
//     files and slower encoding. Requires an ffmpeg built with libvpx-vp9
//     (any --enable-libvpx build has both VP8 and VP9); falls back to lossy
//     VP8 WebM, then to the PNG frame-stream protocol, when unavailable.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/x/sdcpp"
)

// maxWebMOutputBytes caps how much encoded WebM output EncodeWebM will hold
// in memory. This is a safety limit against pathological frame-count/size
// combinations producing an unbounded response; it is intentionally generous
// since WebM/VP8 output is far smaller than the raw RGB input for any
// reasonable video length, and legitimate outputs are expected to be well
// under this cap. It is a variable (not a const) so tests can lower it
// temporarily to exercise the overflow path without generating gigabytes of
// video.
var maxWebMOutputBytes int64 = 512 * 1024 * 1024 // 512 MiB

var (
	ffmpegPathOnce sync.Once
	ffmpegPath     string
	ffmpegPathErr  error

	vp9ProbeFunc   func() bool
	vp9SupportOnce *sync.Once
	vp9Supported   bool
)

func init() {
	vp9ProbeFunc = probeVP9Impl
	vp9SupportOnce = &sync.Once{}
}

// lookupFFmpeg resolves and caches the ffmpeg binary path from PATH.
func lookupFFmpeg() (string, error) {
	ffmpegPathOnce.Do(func() {
		ffmpegPath, ffmpegPathErr = exec.LookPath("ffmpeg")
	})
	return ffmpegPath, ffmpegPathErr
}

// probeVP9Impl is the real VP9 codec probe. It is wrapped by probeVP9 so
// tests can substitute a stub via vp9ProbeFunc.
func probeVP9Impl() bool {
	ffmpeg, err := lookupFFmpeg()
	if err != nil {
		return false
	}
	out, err := exec.Command(ffmpeg, "-hide_banner", "-encoders").CombinedOutput()
	if err != nil {
		return false
	}
	// ffmpeg -encoders lists lines like " V....D libvpx-vp9 ...".
	// A libvpx build that lacks VP9 (unusual but possible via a
	// custom build) would not list libvpx-vp9.
	return strings.Contains(string(out), "libvpx-vp9")
}

// probeVP9 reports whether the resolved ffmpeg was built with the libvpx-vp9
// encoder (required for lossless VP9 output). Any ffmpeg built with
// --enable-libvpx exposes both VP8 (libvpx) and VP9 (libvpx-vp9); builds
// without libvpx or with a stripped encoder list report VP9 as unavailable,
// and lossless requests fall back to lossy VP8.
func probeVP9() bool {
	vp9SupportOnce.Do(func() {
		vp9Supported = vp9ProbeFunc()
	})
	return vp9Supported
}

// SupportsContainerEncoding reports whether an ffmpeg binary is available on
// PATH for WebM container muxing. Callers should fall back to the PNG
// frame-stream protocol when this returns false.
func SupportsContainerEncoding() bool {
	_, err := lookupFFmpeg()
	return err == nil
}

// SupportsLosslessVP9 reports whether the resolved ffmpeg can encode lossless
// VP9 (libvpx-vp9). Implies SupportsContainerEncoding. Callers requesting
// lossless VP9 should fall back to lossy VP8 WebM, then to the PNG
// frame-stream protocol, when this returns false.
func SupportsLosslessVP9() bool {
	return probeVP9()
}

// webmCodec selects the ffmpeg encoder and flags for a given output format.
// Returns the codec name, the extra ffmpeg argument fragments (appended after
// the shared rawvideo input flags), and the container "format" label reported
// to callers. The lossless boolean selects VP9 lossless (libvpx-vp9) over VP8
// (libvpx).
func webmArgs(lossless bool) (codec string, extra []string, format string) {
	if lossless {
		return "libvpx-vp9", []string{"-lossless", "1", "-b:v", "0", "-deadline", "good"}, "webm"
	}
	return "libvpx", []string{"-b:v", "2M", "-deadline", "good"}, "webm"
}

// EncodeWebM muxes a sequence of same-sized raw RGB frames into a WebM
// container at the given frame rate by streaming them through an external
// ffmpeg process (rawvideo in, webm out). When lossless is true, the VP9
// lossless encoder (libvpx-vp9 -lossless 1) is used for pixel-perfect output;
// otherwise the VP8 encoder (libvpx) is used. It returns an error, with no
// partial output, if ffmpeg is unavailable, the requested VP9 codec is not
// compiled in (for lossless), the frames are invalid, or encoding fails for
// any other reason; callers should fall back to the PNG frame-stream protocol
// (or, for lossless failures specifically, to lossy VP8) in that case.
func EncodeWebM(ctx context.Context, frames []sdcpp.Image, fps int, lossless bool) ([]byte, error) {
	if len(frames) == 0 {
		return nil, errors.New("no frames to encode")
	}
	ffmpeg, err := lookupFFmpeg()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found on PATH: %w", err)
	}
	if lossless && !probeVP9() {
		return nil, fmt.Errorf("ffmpeg does not provide the libvpx-vp9 encoder required for lossless VP9")
	}
	if fps <= 0 {
		fps = 16
	}

	w, h := frames[0].Width, frames[0].Height
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("invalid frame dimensions %dx%d", w, h)
	}
	frameSize := w * h * 3
	for i, f := range frames {
		if f.Channel != 3 {
			return nil, fmt.Errorf("frame %d: expected 3-channel RGB, got %d channels", i, f.Channel)
		}
		if f.Width != w || f.Height != h {
			return nil, fmt.Errorf("frame %d: dimension mismatch %dx%d, want %dx%d", i, f.Width, f.Height, w, h)
		}
		if len(f.Data) < frameSize {
			return nil, fmt.Errorf("frame %d: data too short: got %d bytes, want %d", i, len(f.Data), frameSize)
		}
	}

	codec, extraArgs, _ := webmArgs(lossless)
	pixFmt := "yuv420p"
	if lossless {
		// VP9 lossless requires yuv444p to avoid chroma subsampling;
		// yuv420p would still lose color information even with
		// -lossless 1.
		pixFmt = "yuv444p"
	}
	args := []string{
		"-hide_banner", "-loglevel", "error", "-y",
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"-s", fmt.Sprintf("%dx%d", w, h),
		"-r", strconv.Itoa(fps),
		"-i", "-",
		"-an",
		"-c:v", codec,
		"-pix_fmt", pixFmt,
	}
	args = append(args, extraArgs...)
	args = append(args, "-f", "webm", "-")
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

	// Frames are written to stdin on a separate goroutine so that reading
	// stdout below (which must happen concurrently for ffmpeg to make
	// progress on both ends of the pipeline without deadlocking once an OS
	// pipe buffer fills) can proceed independently.
	writeDone := make(chan error, 1)
	go func() {
		defer stdin.Close()
		for _, f := range frames {
			if _, err := stdin.Write(f.Data[:frameSize]); err != nil {
				writeDone <- err
				return
			}
		}
		writeDone <- nil
	}()

	// Fully drain stdout (bounded by maxWebMOutputBytes) before calling
	// Wait(): per the os/exec StdoutPipe contract, Wait closes the pipe once
	// it sees the process exit, so it is incorrect to call Wait before all
	// reads from the pipe have completed — doing so risks truncating the
	// read with a "file already closed" error.
	var buf bytes.Buffer
	n, readErr := io.CopyN(&buf, stdout, maxWebMOutputBytes+1)
	if errors.Is(readErr, io.EOF) {
		readErr = nil
	}
	oversized := n > maxWebMOutputBytes
	if oversized {
		// Drain and discard the remainder so ffmpeg can finish writing
		// without blocking on a full pipe, without retaining it in memory.
		io.Copy(io.Discard, stdout)
	}

	runErr := cmd.Wait()
	writeErr := <-writeDone

	if runErr != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return nil, fmt.Errorf("ffmpeg webm encoding failed: %s", msg)
	}
	if writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
		return nil, fmt.Errorf("failed to stream frames to ffmpeg: %w", writeErr)
	}
	if readErr != nil {
		return nil, fmt.Errorf("failed to read ffmpeg output: %w", readErr)
	}
	if oversized {
		return nil, fmt.Errorf("ffmpeg output exceeded %d byte limit", maxWebMOutputBytes)
	}
	if buf.Len() == 0 {
		return nil, errors.New("ffmpeg produced no output")
	}
	return buf.Bytes(), nil
}
