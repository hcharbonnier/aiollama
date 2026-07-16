package diffgen

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/x/diffutil"
)

// Options holds generation options for image and video.
type Options struct {
	Width          int
	Height         int
	Steps          int
	Seed           int
	NegativePrompt string

	CFGScale     float32
	Sampler      string
	OutputFormat string

	VideoFrames int
	FPS         int
	FlowShift   float32

	InitImage string
	EndImage  string
}

// DefaultOptions returns the default diffgen options.
func DefaultOptions() Options {
	return Options{
		Width:        defaultWidth,
		Height:       defaultHeight,
		Steps:        defaultSteps,
		Seed:         defaultSeed,
		OutputFormat: "png",
	}
}

// RunCLI handles the CLI for diffgen models (image and video).
func RunCLI(cmd *cobra.Command, name string, prompt string, interactive bool, keepAlive *api.Duration) error {
	opts := DefaultOptions()
	readFlags(cmd, &opts)

	if interactive {
		return runInteractive(cmd, name, keepAlive, opts)
	}
	return generate(cmd, name, prompt, keepAlive, opts)
}

func readFlags(cmd *cobra.Command, opts *Options) {
	if cmd == nil || cmd.Flags() == nil {
		return
	}
	if v, err := cmd.Flags().GetInt("width"); err == nil && v > 0 {
		opts.Width = v
	}
	if v, err := cmd.Flags().GetInt("height"); err == nil && v > 0 {
		opts.Height = v
	}
	if v, err := cmd.Flags().GetInt("steps"); err == nil && v > 0 {
		opts.Steps = v
	}
	if v, err := cmd.Flags().GetInt("seed"); err == nil && v != 0 {
		opts.Seed = v
	}
	if v, err := cmd.Flags().GetString("negative"); err == nil && v != "" {
		opts.NegativePrompt = v
	}
	if v, err := cmd.Flags().GetFloat32("cfg-scale"); err == nil && v > 0 {
		opts.CFGScale = v
	}
	if v, err := cmd.Flags().GetString("sampler"); err == nil && v != "" {
		opts.Sampler = v
	}
	if v, err := cmd.Flags().GetString("output-format"); err == nil && v != "" {
		opts.OutputFormat = v
	}
	if v, err := cmd.Flags().GetInt("video-frames"); err == nil && v > 0 {
		opts.VideoFrames = v
	}
	if v, err := cmd.Flags().GetInt("fps"); err == nil && v > 0 {
		opts.FPS = v
	}
	if v, err := cmd.Flags().GetFloat32("flow-shift"); err == nil && v > 0 {
		opts.FlowShift = v
	}
	if v, err := cmd.Flags().GetString("init-image"); err == nil && v != "" {
		opts.InitImage = v
	}
	if v, err := cmd.Flags().GetString("end-image"); err == nil && v != "" {
		opts.EndImage = v
	}
}

// buildRequest constructs an api.GenerateRequest from the prompt and options.
// It also collects init/end images from the --init-image/--end-image flags
// and any image file paths embedded in the prompt text.
func buildRequest(modelName, prompt string, opts Options, keepAlive *api.Duration) (*api.GenerateRequest, error) {
	prompt, images, err := diffutil.ExtractFileData(prompt)
	if err != nil {
		return nil, err
	}

	// Prepend the init-image flag so it is the first image (img2img / I2V).
	if opts.InitImage != "" {
		data, err := diffutil.GetImageData(opts.InitImage)
		if err != nil {
			return nil, fmt.Errorf("init-image: %w", err)
		}
		images = append([]api.ImageData{data}, images...)
	}

	// End frame for FLF2V: encoded as a separate field on the request.
	var endImage []byte
	if opts.EndImage != "" {
		data, err := diffutil.GetImageData(opts.EndImage)
		if err != nil {
			return nil, fmt.Errorf("end-image: %w", err)
		}
		endImage = data
	}

	req := &api.GenerateRequest{
		Model:          modelName,
		Prompt:         prompt,
		Images:         images,
		Width:          int32(opts.Width),
		Height:         int32(opts.Height),
		Steps:          int32(opts.Steps),
		NegativePrompt: opts.NegativePrompt,
		CFGScale:       opts.CFGScale,
		Sampler:        opts.Sampler,
		OutputFormat:   opts.OutputFormat,
		VideoFrames:    int32(opts.VideoFrames),
		FPS:            int32(opts.FPS),
		FlowShift:      opts.FlowShift,
		EndImage:       endImage,
	}
	if opts.Seed != 0 {
		req.Options = map[string]any{"seed": opts.Seed}
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}
	return req, nil
}

// resultCollector accumulates streamed progress and frames/images from a
// generation call.
type resultCollector struct {
	stepBar           *progress.StepBar
	lastImage         string
	frames            [][]byte
	isVideo           bool
	hasVideoContainer bool
}

func (rc *resultCollector) handle(resp api.GenerateResponse, p *progress.Progress, spinner *progress.Spinner, label string) error {
	if resp.Total > 0 {
		if rc.stepBar == nil {
			spinner.Stop()
			rc.stepBar = progress.NewStepBar(label, int(resp.Total))
			p.Add("", rc.stepBar)
		}
		rc.stepBar.Set(int(resp.Completed))
	}
	if resp.Image != "" {
		rc.lastImage = resp.Image
		if rc.isVideo {
			if data, err := base64.StdEncoding.DecodeString(resp.Image); err == nil {
				rc.frames = append(rc.frames, data)
			}
		}
	}
	if resp.Video != "" {
		rc.lastImage = resp.Video
		rc.hasVideoContainer = true
	}
	return nil
}

func generate(cmd *cobra.Command, modelName, prompt string, keepAlive *api.Duration, opts Options) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	req, err := buildRequest(modelName, prompt, opts, keepAlive)
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	label := "Generating"
	if opts.VideoFrames > 0 {
		label = "Generating video"
	}
	rc := &resultCollector{isVideo: opts.VideoFrames > 0}
	err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
		return rc.handle(resp, p, spinner, label)
	})

	p.StopAndClear()
	if err != nil {
		return err
	}

	return saveResult(rc, opts, prompt)
}

func runInteractive(cmd *cobra.Command, modelName string, keepAlive *api.Duration, opts Options) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)
	preloadReq := &api.GenerateRequest{Model: modelName, KeepAlive: keepAlive}
	if err := client.Generate(cmd.Context(), preloadReq, func(api.GenerateResponse) error { return nil }); err != nil {
		p.StopAndClear()
		return fmt.Errorf("failed to load model: %w", err)
	}
	p.StopAndClear()

	scanner, err := readline.New(readline.Prompt{
		Prompt:      ">>> ",
		Placeholder: "Describe an image or video to generate (/help for commands)",
	})
	if err != nil {
		return err
	}

	if envconfig.NoHistory() {
		scanner.HistoryDisable()
	}

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}
			continue
		case err != nil:
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		switch {
		case strings.HasPrefix(line, "/bye"):
			return nil
		case strings.HasPrefix(line, "/?"), strings.HasPrefix(line, "/help"):
			printInteractiveHelp()
			continue
		case strings.HasPrefix(line, "/set "):
			if err := handleSetCommand(line[5:], &opts); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			}
			continue
		case strings.HasPrefix(line, "/show"):
			printCurrentSettings(opts)
			continue
		case strings.HasPrefix(line, "/"):
			args := strings.Fields(line)
			isFile := false
			for _, f := range diffutil.ExtractFileNames(line) {
				if strings.HasPrefix(f, args[0]) {
					isFile = true
					break
				}
			}
			if !isFile {
				fmt.Fprintf(os.Stderr, "Unknown command: %s (try /help)\n", args[0])
				continue
			}
		}

		req, err := buildRequest(modelName, line, opts, keepAlive)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		p := progress.NewProgress(os.Stderr)
		spinner := progress.NewSpinner("")
		p.Add("", spinner)
		label := "Generating"
		if opts.VideoFrames > 0 {
			label = "Generating video"
		}
		rc := &resultCollector{isVideo: opts.VideoFrames > 0}
		err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
			return rc.handle(resp, p, spinner, label)
		})
		p.StopAndClear()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		if err := saveResult(rc, opts, line); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		fmt.Println()
	}
}

// saveResult writes the final image or video to disk. For video, frames
// streamed as base64 PNGs are assembled into a single container (GIF by
// default; falls back to a PNG of the first frame if encoding fails).
func saveResult(rc *resultCollector, opts Options, prompt string) error {
	if rc.lastImage == "" {
		return nil
	}

	safeName := diffutil.SanitizeFilename(prompt)
	if len(safeName) > 50 {
		safeName = safeName[:50]
	}
	timestamp := time.Now().Format("20060102-150405")

	// Video container response (resp.Video was set): write the container
	// bytes directly with the appropriate extension.
	if rc.hasVideoContainer {
		ext := "." + cmp.Or(opts.OutputFormat, "webm")
		filename := fmt.Sprintf("%s-%s%s", safeName, timestamp, ext)
		data, err := base64.StdEncoding.DecodeString(rc.lastImage)
		if err != nil {
			return fmt.Errorf("failed to decode video: %w", err)
		}
		if err := os.WriteFile(filename, data, 0o644); err != nil {
			return fmt.Errorf("failed to save video: %w", err)
		}
		fmt.Printf("Video saved to: %s\n", filename)
		return nil
	}

	// Video frame-stream mode: assemble frames into a GIF container.
	if rc.isVideo && len(rc.frames) > 1 {
		return saveVideo(rc.frames, opts, safeName, timestamp)
	}

	// Single image (or single-frame video treated as an image). resp.Image
	// is always a base64-encoded PNG, so the extension is always .png.
	ext := ".png"
	if rc.isVideo {
		ext = "-video.png"
	}
	filename := fmt.Sprintf("%s-%s%s", safeName, timestamp, ext)
	imageData, err := base64.StdEncoding.DecodeString(rc.lastImage)
	if err != nil {
		return fmt.Errorf("failed to decode image: %w", err)
	}
	if err := os.WriteFile(filename, imageData, 0o644); err != nil {
		return fmt.Errorf("failed to save image: %w", err)
	}
	diffutil.DisplayImageInTerminal(filename)
	fmt.Printf("Image saved to: %s\n", filename)
	return nil
}

// saveVideo encodes collected frames into a container file. GIF is the default
// because it is pure-Go and dependency-free. WebM support is deferred to a
// later phase. Frames are stored as compressed PNG bytes and decoded lazily
// during encoding to minimize peak memory usage.
func saveVideo(frameData [][]byte, opts Options, safeName, timestamp string) error {
	fps := opts.FPS
	if fps <= 0 {
		fps = 16
	}
	delay := 100 / fps
	if delay < 2 {
		delay = 2
	}

	palette := defaultPalette()
	ext := ".gif"
	var buf bytes.Buffer
	if err := encodeGIF(&buf, frameData, delay, palette); err != nil {
		// Fallback: write the first frame as a PNG.
		ext = ".png"
		buf.Reset()
		img, err := decodeImage(frameData[0])
		if err != nil {
			return fmt.Errorf("failed to decode fallback frame: %w", err)
		}
		if err := png.Encode(&buf, img); err != nil {
			return fmt.Errorf("failed to encode fallback PNG: %w", err)
		}
	}

	filename := fmt.Sprintf("%s-%s%s", safeName, timestamp, ext)
	if err := os.WriteFile(filename, buf.Bytes(), 0o644); err != nil {
		return fmt.Errorf("failed to save video: %w", err)
	}
	fmt.Printf("Video saved to: %s (%d frames, %d fps)\n", filename, len(frameData), fps)
	return nil
}

// encodeGIF builds a GIF from compressed PNG frame bytes, decoding each frame
// lazily and quantizing it onto a shared palette using draw.Draw for efficient
// direct blitting.
func encodeGIF(w io.Writer, frameData [][]byte, delay int, palette color.Palette) error {
	out := &gif.GIF{}
	for _, data := range frameData {
		img, err := decodeImage(data)
		if err != nil {
			return err
		}
		bounds := img.Bounds()
		paletted := image.NewPaletted(bounds, palette)
		draw.Draw(paletted, bounds, img, bounds.Min, draw.Src)
		out.Image = append(out.Image, paletted)
		out.Delay = append(out.Delay, delay)
	}
	return gif.EncodeAll(w, out)
}

// decodeImage decodes compressed image bytes (PNG or JPEG) into an image.Image.
func decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	return img, err
}

// defaultPalette returns a 256-color palette suitable for GIF encoding. Uses
// a uniform RGB cube to cover color space reasonably for photographic content.
func defaultPalette() color.Palette {
	p := make(color.Palette, 256)
	idx := 0
	for r := 0; r < 6; r++ {
		for g := 0; g < 7; g++ {
			for b := 0; b < 6; b++ {
				p[idx] = color.RGBA{
					R: uint8(r * 255 / 5),
					G: uint8(g * 255 / 6),
					B: uint8(b * 255 / 5),
					A: 255,
				}
				idx++
			}
		}
	}
	for i := idx; i < 256; i++ {
		v := uint8(float64(i) * 255 / 255)
		p[i] = color.RGBA{R: v, G: v, B: v, A: 255}
	}
	return p
}

func printInteractiveHelp() {
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "  /set width <n>       Set image/video width")
	fmt.Fprintln(os.Stderr, "  /set height <n>      Set image/video height")
	fmt.Fprintln(os.Stderr, "  /set steps <n>       Set denoising steps")
	fmt.Fprintln(os.Stderr, "  /set seed <n>        Set random seed")
	fmt.Fprintln(os.Stderr, "  /set negative <s>    Set negative prompt")
	fmt.Fprintln(os.Stderr, "  /set cfg_scale <f>   Set CFG scale")
	fmt.Fprintln(os.Stderr, "  /set frames <n>      Set video frame count")
	fmt.Fprintln(os.Stderr, "  /set fps <n>         Set video FPS")
	fmt.Fprintln(os.Stderr, "  /set flow_shift <f>  Set WAN flow shift")
	fmt.Fprintln(os.Stderr, "  /set format <s>      Set output format")
	fmt.Fprintln(os.Stderr, "  /show                Show current settings")
	fmt.Fprintln(os.Stderr, "  /bye                 Exit")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Or type a prompt to generate an image or video.")
	fmt.Fprintln(os.Stderr)
}

func printCurrentSettings(opts Options) {
	fmt.Fprintf(os.Stderr, "Current settings:\n")
	fmt.Fprintf(os.Stderr, "  width:    %d\n", opts.Width)
	fmt.Fprintf(os.Stderr, "  height:   %d\n", opts.Height)
	fmt.Fprintf(os.Stderr, "  steps:    %d\n", opts.Steps)
	fmt.Fprintf(os.Stderr, "  seed:     %d (0=random)\n", opts.Seed)
	fmt.Fprintf(os.Stderr, "  format:   %s\n", opts.OutputFormat)
	if opts.CFGScale > 0 {
		fmt.Fprintf(os.Stderr, "  cfg:      %.2f\n", opts.CFGScale)
	}
	if opts.VideoFrames > 0 {
		fmt.Fprintf(os.Stderr, "  frames:   %d\n", opts.VideoFrames)
		fmt.Fprintf(os.Stderr, "  fps:      %d\n", opts.FPS)
	}
	if opts.FlowShift > 0 {
		fmt.Fprintf(os.Stderr, "  flow:     %.2f\n", opts.FlowShift)
	}
	if opts.NegativePrompt != "" {
		fmt.Fprintf(os.Stderr, "  negative: %s\n", opts.NegativePrompt)
	}
	fmt.Fprintln(os.Stderr)
}

func handleSetCommand(args string, opts *Options) error {
	parts := strings.SplitN(args, " ", 2)
	if len(parts) < 2 {
		return fmt.Errorf("usage: /set <option> <value>")
	}
	key := strings.ToLower(parts[0])
	value := strings.TrimSpace(parts[1])
	switch key {
	case "width", "w":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("width must be a positive integer")
		}
		opts.Width = v
		fmt.Fprintf(os.Stderr, "Set width to %d\n", v)
	case "height", "h":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("height must be a positive integer")
		}
		opts.Height = v
		fmt.Fprintf(os.Stderr, "Set height to %d\n", v)
	case "steps", "s":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("steps must be a positive integer")
		}
		opts.Steps = v
		fmt.Fprintf(os.Stderr, "Set steps to %d\n", v)
	case "seed":
		v, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("seed must be an integer")
		}
		opts.Seed = v
		fmt.Fprintf(os.Stderr, "Set seed to %d\n", v)
	case "frames":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("frames must be a positive integer")
		}
		opts.VideoFrames = v
		fmt.Fprintf(os.Stderr, "Set video frames to %d\n", v)
	case "fps":
		v, err := strconv.Atoi(value)
		if err != nil || v <= 0 {
			return fmt.Errorf("fps must be a positive integer")
		}
		opts.FPS = v
		fmt.Fprintf(os.Stderr, "Set fps to %d\n", v)
	case "flow_shift", "flow":
		v, err := strconv.ParseFloat(value, 32)
		if err != nil || v <= 0 {
			return fmt.Errorf("flow_shift must be a positive number")
		}
		opts.FlowShift = float32(v)
		fmt.Fprintf(os.Stderr, "Set flow_shift to %.2f\n", v)
	case "cfg_scale", "cfg":
		v, err := strconv.ParseFloat(value, 32)
		if err != nil || v <= 0 {
			return fmt.Errorf("cfg_scale must be a positive number")
		}
		opts.CFGScale = float32(v)
		fmt.Fprintf(os.Stderr, "Set cfg_scale to %.2f\n", v)
	case "format", "output_format":
		opts.OutputFormat = value
		fmt.Fprintf(os.Stderr, "Set output format to %s\n", value)
	case "negative", "neg", "n":
		opts.NegativePrompt = value
		if value == "" {
			fmt.Fprintln(os.Stderr, "Cleared negative prompt")
		} else {
			fmt.Fprintf(os.Stderr, "Set negative prompt to: %s\n", value)
		}
	default:
		return fmt.Errorf("unknown option: %s (try /help)", key)
	}
	return nil
}
