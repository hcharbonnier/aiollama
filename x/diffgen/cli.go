//go:build sdcpp

package diffgen

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
)

// Options holds generation options for image and video.
type Options struct {
	Width          int
	Height         int
	Steps          int
	Seed           int
	NegativePrompt string

	CFGScale float32
	Sampler  string
	Format   string

	VideoFrames int
	FPS         int
	FlowShift   float32
}

func DefaultOptions() Options {
	return Options{
		Width:  1024,
		Height: 1024,
		Steps:  0,
		Seed:   0,
		Format: "png",
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
	if v, err := cmd.Flags().GetInt("video-frames"); err == nil && v > 0 {
		opts.VideoFrames = v
	}
	if v, err := cmd.Flags().GetInt("fps"); err == nil && v > 0 {
		opts.FPS = v
	}
	if v, err := cmd.Flags().GetFloat32("flow-shift"); err == nil && v > 0 {
		opts.FlowShift = v
	}
}

func generate(cmd *cobra.Command, modelName, prompt string, keepAlive *api.Duration, opts Options) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	prompt, images, err := extractFileData(prompt)
	if err != nil {
		return err
	}

	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Images: images,
		Width:  int32(opts.Width),
		Height: int32(opts.Height),
		Steps:  int32(opts.Steps),
	}
	if opts.Seed != 0 {
		req.Options = map[string]any{"seed": opts.Seed}
	}
	if keepAlive != nil {
		req.KeepAlive = keepAlive
	}

	p := progress.NewProgress(os.Stderr)
	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	var stepBar *progress.StepBar
	var lastImage string
	var frameCount int
	err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
		if resp.Total > 0 {
			if stepBar == nil {
				spinner.Stop()
				label := "Generating"
				if opts.VideoFrames > 0 {
					label = "Generating video"
				}
				stepBar = progress.NewStepBar(label, int(resp.Total))
				p.Add("", stepBar)
			}
			stepBar.Set(int(resp.Completed))
		}
		if resp.Image != "" {
			lastImage = resp.Image
			frameCount++
		}
		return nil
	})

	p.StopAndClear()
	if err != nil {
		return err
	}

	if lastImage != "" {
		safeName := sanitizeFilename(prompt)
		if len(safeName) > 50 {
			safeName = safeName[:50]
		}
		timestamp := time.Now().Format("20060102-150405")
		ext := ".png"
		if frameCount > 1 {
			ext = "-video.png"
		}
		filename := fmt.Sprintf("%s-%s%s", safeName, timestamp, ext)
		imageData, err := base64.StdEncoding.DecodeString(lastImage)
		if err != nil {
			return fmt.Errorf("failed to decode image: %w", err)
		}
		if err := os.WriteFile(filename, imageData, 0o644); err != nil {
			return fmt.Errorf("failed to save image: %w", err)
		}
		displayImageInTerminal(filename)
		fmt.Printf("Image saved to: %s\n", filename)
	}
	return nil
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
			// Check if it's a file path, not a command
			args := strings.Fields(line)
			isFile := false
			for _, f := range extractFileNames(line) {
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

		prompt, images, err := extractFileData(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		req := &api.GenerateRequest{
			Model:  modelName,
			Prompt: prompt,
			Images: images,
			Width:  int32(opts.Width),
			Height: int32(opts.Height),
			Steps:  int32(opts.Steps),
		}
		if opts.Seed != 0 {
			req.Options = map[string]any{"seed": opts.Seed}
		}
		if keepAlive != nil {
			req.KeepAlive = keepAlive
		}

		p := progress.NewProgress(os.Stderr)
		spinner := progress.NewSpinner("")
		p.Add("", spinner)
		var stepBar *progress.StepBar
		var lastImage string
		var frameCount int
		err = client.Generate(cmd.Context(), req, func(resp api.GenerateResponse) error {
			if resp.Total > 0 {
				if stepBar == nil {
					spinner.Stop()
					label := "Generating"
					if opts.VideoFrames > 0 {
						label = "Generating video"
					}
					stepBar = progress.NewStepBar(label, int(resp.Total))
					p.Add("", stepBar)
				}
				stepBar.Set(int(resp.Completed))
			}
			if resp.Image != "" {
				lastImage = resp.Image
				frameCount++
			}
			return nil
		})
		p.StopAndClear()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		if lastImage != "" {
			imageData, err := base64.StdEncoding.DecodeString(lastImage)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error decoding image: %v\n", err)
				continue
			}
			safeName := sanitizeFilename(line)
			if len(safeName) > 50 {
				safeName = safeName[:50]
			}
			timestamp := time.Now().Format("20060102-150405")
			ext := ".png"
			if frameCount > 1 {
				ext = "-video.png"
			}
			filename := fmt.Sprintf("%s-%s%s", safeName, timestamp, ext)
			if err := os.WriteFile(filename, imageData, 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "Error saving image: %v\n", err)
				continue
			}
			displayImageInTerminal(filename)
			fmt.Printf("Image saved to: %s\n", filename)
		}
		fmt.Println()
	}
}

func sanitizeFilename(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, " ", "-")
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func printInteractiveHelp() {
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "  /set width <n>       Set image width")
	fmt.Fprintln(os.Stderr, "  /set height <n>      Set image height")
	fmt.Fprintln(os.Stderr, "  /set steps <n>       Set denoising steps")
	fmt.Fprintln(os.Stderr, "  /set seed <n>        Set random seed")
	fmt.Fprintln(os.Stderr, "  /set negative <s>    Set negative prompt")
	fmt.Fprintln(os.Stderr, "  /set frames <n>      Set video frame count")
	fmt.Fprintln(os.Stderr, "  /set fps <n>          Set video FPS")
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
	if opts.VideoFrames > 0 {
		fmt.Fprintf(os.Stderr, "  frames:   %d\n", opts.VideoFrames)
		fmt.Fprintf(os.Stderr, "  fps:      %d\n", opts.FPS)
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

func displayImageInTerminal(imagePath string) bool {
	termProgram := os.Getenv("TERM_PROGRAM")
	kittyWindowID := os.Getenv("KITTY_WINDOW_ID")
	weztermPane := os.Getenv("WEZTERM_PANE")
	ghostty := os.Getenv("GHOSTTY_RESOURCES_DIR")

	data, err := os.ReadFile(imagePath)
	if err != nil {
		return false
	}
	encoded := base64.StdEncoding.EncodeToString(data)

	switch {
	case termProgram == "iTerm.app" || termProgram == "WezTerm" || weztermPane != "":
		fmt.Printf("\033]1337;File=inline=1;preserveAspectRatio=1:%s\a\n", encoded)
		return true
	case kittyWindowID != "" || ghostty != "" || termProgram == "ghostty":
		const chunkSize = 4096
		for i := 0; i < len(encoded); i += chunkSize {
			end := min(i+chunkSize, len(encoded))
			chunk := encoded[i:end]
			if i == 0 {
				more := 1
				if end >= len(encoded) {
					more = 0
				}
				fmt.Printf("\033_Ga=T,f=100,m=%d;%s\033\\", more, chunk)
			} else if end >= len(encoded) {
				fmt.Printf("\033_Gm=0;%s\033\\", chunk)
			} else {
				fmt.Printf("\033_Gm=1;%s\033\\", chunk)
			}
		}
		fmt.Println()
		return true
	default:
		return false
	}
}

func extractFileNames(input string) []string {
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp)\b`
	re := regexp.MustCompile(regexPattern)
	return re.FindAllString(input, -1)
}

func extractFileData(input string) (string, []api.ImageData, error) {
	filePaths := extractFileNames(input)
	var imgs []api.ImageData
	for _, fp := range filePaths {
		nfp := strings.ReplaceAll(fp, "\\ ", " ")
		nfp = strings.ReplaceAll(nfp, "\\(", "(")
		nfp = strings.ReplaceAll(nfp, "\\)", ")")
		nfp = strings.ReplaceAll(nfp, "%20", " ")
		data, err := getImageData(nfp)
		if errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return "", nil, err
		}
		fmt.Fprintf(os.Stderr, "Added image '%s'\n", nfp)
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, data)
	}
	return strings.TrimSpace(input), imgs, nil
}

func getImageData(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	buf := make([]byte, 512)
	if _, err = file.Read(buf); err != nil {
		return nil, err
	}
	contentType := http.DetectContentType(buf)
	allowedTypes := []string{"image/jpeg", "image/jpg", "image/png", "image/webp"}
	if !slices.Contains(allowedTypes, contentType) {
		return nil, fmt.Errorf("invalid image type: %s", contentType)
	}
	return os.ReadFile(filePath)
}
