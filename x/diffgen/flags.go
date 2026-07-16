package diffgen

import (
	"github.com/spf13/cobra"
)

// flagDefaults mirrors the defaults in DefaultOptions so RegisterFlags and
// DefaultOptions stay in sync.
const (
	defaultWidth  = 1024
	defaultHeight = 1024
	defaultSteps  = 0
	defaultSeed   = 0
)

// RegisterFlags adds diffgen (image + video) generation flags to the given
// command. Flags are hidden since they only apply to diffusion models.
func RegisterFlags(cmd *cobra.Command) {
	cmd.Flags().Int("width", defaultWidth, "Image/video width")
	cmd.Flags().Int("height", defaultHeight, "Image/video height")
	cmd.Flags().Int("steps", defaultSteps, "Denoising steps (0 = model default)")
	cmd.Flags().Int("seed", defaultSeed, "Random seed (0 for random)")
	cmd.Flags().String("negative", "", "Negative prompt")
	cmd.Flags().Float32("cfg-scale", 0, "Classifier-free guidance scale")
	cmd.Flags().String("sampler", "", "Sampler name (e.g. euler)")
	cmd.Flags().String("output-format", "", "Output format (image: png; video: gif, webm, webp)")

	// Video-specific flags
	cmd.Flags().Int("video-frames", 0, "Number of video frames to generate")
	cmd.Flags().Int("fps", 0, "Output frame rate for video")
	cmd.Flags().Float32("flow-shift", 0, "Flow shift parameter for WAN video models")

	// Image-to-image / image-to-video flags
	cmd.Flags().String("init-image", "", "Path to init image (img2img / I2V)")
	cmd.Flags().String("end-image", "", "Path to end frame image (FLF2V video)")

	for _, f := range []string{
		"width", "height", "steps", "seed", "negative", "cfg-scale",
		"sampler", "output-format", "video-frames", "fps", "flow-shift",
		"init-image", "end-image",
	} {
		_ = cmd.Flags().MarkHidden(f)
	}
}

// AppendFlagsDocs appends diffgen flag documentation to the command's usage
// template.
func AppendFlagsDocs(cmd *cobra.Command) {
	usage := `
Diffusion Generation Flags (image + video, experimental):
      --width int          Image/video width
      --height int         Image/video height
      --steps int          Denoising steps (0 = model default)
      --seed int           Random seed (0 for random)
      --negative str       Negative prompt
      --cfg-scale float     Classifier-free guidance scale
      --sampler str         Sampler name (e.g. euler)
      --output-format str   Output format (image: png; video: gif, webm, webp)
      --video-frames int    Number of video frames to generate
      --fps int             Output frame rate for video
      --flow-shift float    Flow shift for WAN video models
      --init-image str      Path to init image (img2img / I2V)
      --end-image str        Path to end frame image (FLF2V video)
`
	cmd.SetUsageTemplate(cmd.UsageTemplate() + usage)
}
