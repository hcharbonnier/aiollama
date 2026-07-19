package videojobs

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
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

// maxDecodedFrames bounds how many frames DecodeFrames will extract from a
// source video. This prevents a pathological source (e.g. a 2-hour movie) from
// exhausting memory; the edit/extend handlers only need the first or last
// frame(s) anyway. It is a variable (not a const) so tests can lower it.
var maxDecodedFrames int = 120

// DecodeFrames extracts PNG-encoded frames from an MP4 (or any ffmpeg-readable
// container) by piping it through ffmpeg as image2png output. Frames are
// returned in playback order as PNG bytes. maxFrames <= 0 means all frames
// (capped by maxDecodedFrames). The returned int is the fps ffmpeg probed
// from the source container (0 if unknown).
//
// This is the inverse of EncodeMP4 and is used by /v1/videos/edits and
// /v1/videos/extensions: the source video (a previously-generated completed
// job's MP4) is decoded back to frames so the first/last frame can be fed to
// the diffgen runner as an I2V/V2V init image.
func (t *ffmpegTranscoder) DecodeFrames(ctx context.Context, mp4 []byte, maxFrames int) ([][]byte, int, error) {
	if len(mp4) == 0 {
		return nil, 0, errors.New("no video to decode")
	}
	ffmpeg, err := t.lookup()
	if err != nil {
		return nil, 0, fmt.Errorf("ffmpeg not found on PATH: %w", err)
	}
	if maxFrames <= 0 || maxFrames > maxDecodedFrames {
		maxFrames = maxDecodedFrames
	}

	// Read the source from stdin and emit PNG frames to stdout via the
	// image2 muxer. -v quiet suppresses the per-frame progress banner.
	args := []string{
		"-hide_banner", "-loglevel", "error",
		"-i", "pipe:0",
		"-frames:v", strconv.Itoa(maxFrames),
		"-f", "image2pipe",
		"-vcodec", "png",
		"pipe:1",
	}
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open ffmpeg stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open ffmpeg stdout: %w", err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, 0, fmt.Errorf("failed to start ffmpeg: %w", err)
	}

	// Write the MP4 bytes to stdin on a goroutine so stdout can be drained
	// concurrently (ffmpeg reads input and writes output in parallel).
	writeDone := make(chan error, 1)
	go func() {
		defer stdin.Close()
		_, err := io.Copy(stdin, bytes.NewReader(mp4))
		writeDone <- err
	}()

	// Split the PNG stream into individual frames. The image2pipe muxer
	// concatenates PNGs; png.Decode reads exactly one image per call.
	frames, fps, readErr := splitPNGStream(stdout, maxFrames)
	runErr := cmd.Wait()
	writeErr := <-writeDone

	if runErr != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return nil, 0, fmt.Errorf("ffmpeg decode failed: %s", msg)
	}
	if writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
		return nil, 0, fmt.Errorf("failed to stream video to ffmpeg: %w", writeErr)
	}
	if readErr != nil && len(frames) == 0 {
		return nil, 0, fmt.Errorf("failed to read ffmpeg frames: %w", readErr)
	}
	return frames, fps, nil
}

// DecodeLastFrame extracts only the final frame of a video as a PNG. It uses
// ffmpeg's -sseof (seek from end of file) with a small negative offset so
// ffmpeg decodes only the last frame, not the whole clip. This avoids the O(n)
// decode cost of DecodeFrames and, critically, returns the TRUE last frame
// regardless of source length (DecodeFrames with -frames:v N returns the first
// N frames, which is wrong for "last frame" when the source exceeds the cap).
//
// Some ffmpeg builds / containers do not support -sseof with pipe input; in
// that case it falls back to decoding all frames and returning the last one
// (bounded by maxDecodedFrames).
func (t *ffmpegTranscoder) DecodeLastFrame(ctx context.Context, mp4 []byte) ([]byte, error) {
	if len(mp4) == 0 {
		return nil, errors.New("no video to decode")
	}
	ffmpeg, err := t.lookup()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found on PATH: %w", err)
	}

	// -sseof -0.1 seeks to 0.1s before the end; -frames:v 1 grabs one frame
	// from there. For pipe input, -sseof may not work (it needs a seekable
	// input); we handle that with the fallback below.
	args := []string{
		"-hide_banner", "-loglevel", "error",
		"-sseof", "-0.1",
		"-i", "pipe:0",
		"-frames:v", "1",
		"-f", "image2pipe",
		"-vcodec", "png",
		"pipe:1",
	}
	frame, err := t.decodeOneFrame(ctx, ffmpeg, args, mp4)
	if err == nil && len(frame) > 0 {
		return frame, nil
	}

	// Fallback: decode all frames (capped) and return the last one. This is
	// the correct but slower path for containers/ffmpeg builds where -sseof
	// doesn't work with pipe input.
	frames, _, decodeErr := t.DecodeFrames(ctx, mp4, maxDecodedFrames)
	if decodeErr != nil && len(frames) == 0 {
		return nil, fmt.Errorf("decode last frame (sseof failed: %v; fallback also failed: %v)", err, decodeErr)
	}
	if len(frames) == 0 {
		return nil, fmt.Errorf("decode last frame: source produced no frames (sseof failed: %v)", err)
	}
	return frames[len(frames)-1], nil
}

// decodeOneFrame runs ffmpeg with the given args, writes mp4 to stdin, and
// reads a single PNG frame from stdout. Returns the PNG bytes or an error.
func (t *ffmpegTranscoder) decodeOneFrame(ctx context.Context, ffmpeg string, args []string, mp4 []byte) ([]byte, error) {
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

	writeDone := make(chan error, 1)
	go func() {
		defer stdin.Close()
		_, err := io.Copy(stdin, bytes.NewReader(mp4))
		writeDone <- err
	}()

	// Read the single PNG frame. Use io.ReadAll since it's one frame.
	data, readErr := io.ReadAll(stdout)
	runErr := cmd.Wait()
	writeErr := <-writeDone

	if runErr != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return nil, fmt.Errorf("ffmpeg decode failed: %s", msg)
	}
	if writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
		return nil, fmt.Errorf("failed to stream video to ffmpeg: %w", writeErr)
	}
	if readErr != nil {
		return nil, fmt.Errorf("failed to read ffmpeg output: %w", readErr)
	}
	if len(data) == 0 {
		return nil, errors.New("ffmpeg produced no output")
	}
	return data, nil
}

// splitPNGStream reads a stream of concatenated PNG images from r, decoding
// each via png.DecodeConfig to discover its boundaries. Returns the slice of
// raw PNG bytes (one per image) and the probed fps (always 0 from image2pipe;
// fps is reported by a separate probe when needed). Stops after max frames or
// at EOF.
func splitPNGStream(r io.Reader, max int) ([][]byte, int, error) {
	// Read the whole stream into memory (bounded by maxFrames of small
	// frames for the edit/extend use case). A streaming chunked reader would
	// be more memory-efficient, but PNG frames do not have a length prefix
	// so we must scan for IEND chunks; buffering then splitting is simplest.
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, 0, err
	}
	var frames [][]byte
	pos := 0
	for pos < len(data) && (max <= 0 || len(frames) < max) {
		cfg, err := png.DecodeConfig(bytes.NewReader(data[pos:]))
		if err != nil {
			// Trailing bytes after the last frame: stop cleanly.
			break
		}
		// Re-encode the frame's bytes by decoding the full image to find
		// its end. png.Decode consumes exactly one image.
		img, err := png.Decode(bytes.NewReader(data[pos:]))
		if err != nil {
			break
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			break
		}
		frames = append(frames, buf.Bytes())
		// Advance past the decoded image: use the re-encoded length as an
		// approximation only if the source isn't a clean stream; in
		// practice image2pipe emits clean concatenated PNGs so we rescan
		// from pos+len(buf). To be robust to minor differences, fall back to
		// scanning for the next PNG signature.
		next := findNextPNGSig(data, pos+1)
		if next < 0 {
			break
		}
		_ = cfg // (cfg retained for future dimension validation)
		pos = next
	}
	return frames, 0, nil
}

// findNextPNGSig returns the index of the next PNG signature (8 bytes:
// \x89PNG\r\n\x1a\n) at or after start, or -1 if none.
func findNextPNGSig(data []byte, start int) int {
	sig := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}
	if start < 0 {
		start = 0
	}
	for i := start; i <= len(data)-len(sig); i++ {
		match := true
		for j := 0; j < len(sig); j++ {
			if data[i+j] != sig[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

// maxConcatInputBytes bounds how much input ConcatMP4 will read into memory.
// Each input is a generated MP4 (bounded by maxMP4OutputBytes), so this is a
// generous safety cap against accidental misuse with huge external files.
var maxConcatInputBytes int64 = 2 * 1024 * 1024 * 1024 // 2 GiB

// ffprobeLookup caches the ffprobe lookup (ships with ffmpeg).
var ffprobeLookup struct {
	once sync.Once
	path string
	err  error
}

func lookupFFprobe() (string, error) {
	ffprobeLookup.once.Do(func() {
		ffprobeLookup.path, ffprobeLookup.err = exec.LookPath("ffprobe")
	})
	return ffprobeLookup.path, ffprobeLookup.err
}

// FFmpegPath returns the resolved ffmpeg binary path, sharing the default
// transcoder's cached lookup. It lets other packages (e.g. the Images API
// WebP transcoder) reuse one ffmpeg resolution instead of duplicating it.
func FFmpegPath() (string, error) {
	return defaultTranscoder.lookup()
}

// FFprobePath returns the resolved ffprobe binary path (cached lookup).
func FFprobePath() (string, error) {
	return lookupFFprobe()
}

// maxProbeFallbackBytes bounds how much of an uploaded video the ffmpeg -i
// duration fallback reads. The container header (which carries the Duration
// line) sits at the start for the MP4s this API handles, so a few MiB
// suffice; feeding the full (up to 256 MiB) upload twice per request would
// be wasteful.
var maxProbeFallbackBytes int64 = 8 << 20 // 8 MiB

// ProbeDurationSeconds returns the video duration in whole seconds, rounded
// to nearest. It prefers ffprobe; if ffprobe is absent or fails (e.g. a
// container it cannot parse from a pipe), it falls back to parsing the
// "Duration: HH:MM:SS.cc" line from ffmpeg -i stderr output.
func (t *ffmpegTranscoder) ProbeDurationSeconds(ctx context.Context, mp4 []byte) (int, error) {
	if len(mp4) == 0 {
		return 0, errors.New("no video to probe")
	}
	if ffprobe, err := lookupFFprobe(); err == nil {
		args := []string{
			"-v", "error",
			"-show_entries", "format=duration",
			"-of", "default=noprint_wrappers=1:nokey=1",
			"pipe:0",
		}
		cmd := exec.CommandContext(ctx, ffprobe, args...)
		cmd.Stdin = bytes.NewReader(mp4)
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if runErr := cmd.Run(); runErr == nil {
			out := strings.TrimSpace(stdout.String())
			if secs, parseErr := strconv.ParseFloat(out, 64); parseErr == nil && secs > 0 {
				return int(secs + 0.5), nil
			}
		}
	}

	// Fallback: ffmpeg -i prints "Duration: 00:00:04.03" on stderr (and exits
	// non-zero because no output was specified — expected). Only the head of
	// the container is needed for the duration line.
	ffmpeg, err := t.lookup()
	if err != nil {
		return 0, fmt.Errorf("ffprobe and ffmpeg not found on PATH: %w", err)
	}
	head := mp4
	if int64(len(head)) > maxProbeFallbackBytes {
		head = head[:maxProbeFallbackBytes]
	}
	cmd := exec.CommandContext(ctx, ffmpeg, "-hide_banner", "-i", "pipe:0")
	cmd.Stdin = bytes.NewReader(head)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	_ = cmd.Run() // non-zero exit is expected (no output file)

	const marker = "Duration: "
	idx := strings.Index(stderr.String(), marker)
	if idx < 0 {
		return 0, fmt.Errorf("could not determine video duration (no Duration line in ffmpeg output)")
	}
	fields := strings.Fields(stderr.String()[idx+len(marker):])
	if len(fields) == 0 {
		return 0, fmt.Errorf("could not parse video duration")
	}
	hms := strings.TrimSuffix(fields[0], ",")
	var hh, mm int
	var ss float64
	if _, err := fmt.Sscanf(hms, "%d:%d:%f", &hh, &mm, &ss); err != nil {
		return 0, fmt.Errorf("could not parse video duration %q: %w", hms, err)
	}
	total := float64(hh*3600+mm*60) + ss
	if total <= 0 {
		return 0, fmt.Errorf("video has non-positive duration %q", hms)
	}
	return int(total + 0.5), nil
}

// Spritesheet renders a tiled grid of frames sampled across the video as a
// single PNG, via ffmpeg's tile filter. Frames are sampled at ~1 fps over
// the clip and tiled 5x5 (25 cells max), scaled to 320px-wide cells — a
// reasonable storyboard for the short clips this API produces.
func (t *ffmpegTranscoder) Spritesheet(ctx context.Context, mp4 []byte) ([]byte, error) {
	if len(mp4) == 0 {
		return nil, errors.New("no video to decode")
	}
	ffmpeg, err := t.lookup()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found on PATH: %w", err)
	}
	args := []string{
		"-hide_banner", "-loglevel", "error",
		"-i", "pipe:0",
		"-vf", "fps=1,scale=320:-2,tile=5x5:margin=2:padding=2",
		"-frames:v", "1",
		"-f", "image2pipe",
		"-vcodec", "png",
		"pipe:1",
	}
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	cmd.Stdin = bytes.NewReader(mp4)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("ffmpeg spritesheet failed: %s", msg)
	}
	if stdout.Len() == 0 {
		return nil, errors.New("ffmpeg spritesheet produced no output")
	}
	return stdout.Bytes(), nil
}

// ConcatMP4 concatenates two MP4 byte streams into a single MP4 using ffmpeg's
// concat demuxer. Both inputs should have been produced by EncodeMP4 (same
// codec/fps). fps sets the output frame rate explicitly to avoid drift. If
// either input is empty, the other is returned unchanged.
//
// Temp files are used because (a) the concat demuxer requires seekable inputs
// (pipes are not seekable) and (b) os/exec.Cmd.ExtraFiles (extra fds) is not
// supported on Windows, so piping two inputs simultaneously into one ffmpeg
// invocation is not portable. The temp files are cleaned up on return.
func (t *ffmpegTranscoder) ConcatMP4(ctx context.Context, first, second []byte, fps int) ([]byte, error) {
	if len(first) == 0 {
		return second, nil
	}
	if len(second) == 0 {
		return first, nil
	}
	if int64(len(first))+int64(len(second)) > maxConcatInputBytes {
		return nil, fmt.Errorf("concat input too large: %d bytes", int64(len(first))+int64(len(second)))
	}
	ffmpeg, err := t.lookup()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found on PATH: %w", err)
	}
	if fps <= 0 {
		fps = 16
	}

	// Write both segments to temp files and build a concat demuxer list.
	dir, err := os.MkdirTemp("", "videojobs-concat-")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	firstPath := filepath.Join(dir, "first.mp4")
	secondPath := filepath.Join(dir, "second.mp4")
	listPath := filepath.Join(dir, "list.txt")
	if err := os.WriteFile(firstPath, first, 0o600); err != nil {
		return nil, fmt.Errorf("write first segment: %w", err)
	}
	if err := os.WriteFile(secondPath, second, 0o600); err != nil {
		return nil, fmt.Errorf("write second segment: %w", err)
	}
	// The concat demuxer list format requires single-quoted, escaped paths.
	list := fmt.Sprintf("file '%s'\nfile '%s'\n", strings.ReplaceAll(firstPath, "'", "'\\''"), strings.ReplaceAll(secondPath, "'", "'\\''"))
	if err := os.WriteFile(listPath, []byte(list), 0o600); err != nil {
		return nil, fmt.Errorf("write concat list: %w", err)
	}

	// Try stream-copy first (-c copy): both segments came from EncodeMP4
	// with the same codec/fps, so a copy concat is far cheaper (no re-encode).
	// If the concat demuxer rejects the copy (mismatched params, fragmented
	// MP4 boundaries), fall back to a full re-encode through libx264.
	result, err := t.concatWithCodec(ctx, ffmpeg, firstPath, secondPath, listPath, fps, true)
	if err == nil && len(result) > 0 {
		return result, nil
	}
	// Fallback: re-encode through libx264.
	return t.concatWithCodec(ctx, ffmpeg, firstPath, secondPath, listPath, fps, false)
}

// concatWithCodec runs the ffmpeg concat demuxer with either stream-copy
// (-c copy, fast) or re-encode (-c:v libx264, slower but always works). The
// temp files and list must already be written. copyOnly selects the mode.
func (t *ffmpegTranscoder) concatWithCodec(ctx context.Context, ffmpeg, firstPath, secondPath, listPath string, fps int, copyOnly bool) ([]byte, error) {
	args := []string{
		"-hide_banner", "-loglevel", "error", "-y",
		"-f", "concat", "-safe", "0",
		"-i", listPath,
	}
	if copyOnly {
		args = append(args, "-c", "copy")
	} else {
		args = append(
			args,
			"-r", strconv.Itoa(fps),
			"-c:v", "libx264",
			"-pix_fmt", "yuv420p",
			"-preset", "veryfast",
			"-crf", "23",
		)
	}
	args = append(
		args,
		"-movflags", "+frag_keyframe+empty_moov+default_base_moof",
		"-f", "mp4", "pipe:1",
	)
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to open ffmpeg stdout: %w", err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start ffmpeg: %w", err)
	}

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

	if runErr != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return nil, fmt.Errorf("ffmpeg concat failed: %s", msg)
	}
	if readErr != nil {
		return nil, fmt.Errorf("failed to read ffmpeg concat output: %w", readErr)
	}
	if oversized {
		return nil, fmt.Errorf("ffmpeg concat output exceeded %d byte limit", maxMP4OutputBytes)
	}
	if buf.Len() == 0 {
		return nil, errors.New("ffmpeg concat produced no output")
	}
	return buf.Bytes(), nil
}
