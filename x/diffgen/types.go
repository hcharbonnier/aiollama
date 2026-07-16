// Package diffgen provides a unified runner subprocess for image and video
// generation via the stable-diffusion.cpp native backend. It replaces the
// former MLX-based x/imagegen package.
//
// The runner implements llm.LlamaServer and is spawned as a subprocess
// (`ollama runner --diffgen-engine`), exposing a local HTTP server with
// /health and /completion handlers that stream ndjson progress + results.
package diffgen

// DiffRequest is the unified request format for both image and video
// generation. Mode is auto-detected from the loaded model's capabilities when
// empty.
type DiffRequest struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"negative_prompt,omitempty"`
	Mode           string `json:"mode,omitempty"` // "image" or "video"; auto if empty

	Width  int32 `json:"width,omitempty"`
	Height int32 `json:"height,omitempty"`
	Steps  int   `json:"steps,omitempty"`
	Seed   int64 `json:"seed,omitempty"`

	CFGScale     float32 `json:"cfg_scale,omitempty"`
	Sampler      string  `json:"sampler,omitempty"`
	Scheduler    string  `json:"scheduler,omitempty"`
	OutputFormat string  `json:"output_format,omitempty"` // image: "png"; video: "webm","webp","gif"
	Images       [][]byte `json:"images,omitempty"`        // init/control images (img2img, I2V)

	BatchCount      int     `json:"batch_count,omitempty"`
	ControlStrength float32 `json:"control_strength,omitempty"`

	VideoFrames int     `json:"video_frames,omitempty"` // e.g. 33
	FPS         int     `json:"fps,omitempty"`          // e.g. 16
	FlowShift   float32 `json:"flow_shift,omitempty"`   // WAN: 3.0
	EndImage    []byte  `json:"end_image,omitempty"`    // FLF2V end frame

	Options *RequestOptions `json:"options,omitempty"`
}

// RequestOptions contains generation options forwarded from the scheduler.
type RequestOptions struct {
	NumPredict  int      `json:"num_predict,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	TopK        int      `json:"top_k,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// DiffResponse is streamed back per progress update and final result.
type DiffResponse struct {
	Content    string `json:"content,omitempty"` // error text
	Image      string `json:"image,omitempty"`   // base64 PNG (image mode)
	Video      string `json:"video,omitempty"`   // base64 container (video mode)
	Done       bool   `json:"done"`
	StopReason string `json:"stop_reason,omitempty"`

	Step  int `json:"step,omitempty"`
	Total int `json:"total,omitempty"`

	Frame  int `json:"frame,omitempty"`  // current frame (video mode)
	Frames int `json:"frames,omitempty"` // total frames (video mode)

	PromptEvalCount    int `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int `json:"prompt_eval_duration,omitempty"`
	EvalCount          int `json:"eval_count,omitempty"`
	EvalDuration       int `json:"eval_duration,omitempty"`
}

// HealthResponse is returned by the /health endpoint.
type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress,omitempty"`
}

// ModelMode represents whether a model generates images or video.
type ModelMode int

const (
	ModeImage ModelMode = iota
	ModeVideo
)

func (m ModelMode) String() string {
	switch m {
	case ModeImage:
		return "image"
	case ModeVideo:
		return "video"
	default:
		return "unknown"
	}
}
