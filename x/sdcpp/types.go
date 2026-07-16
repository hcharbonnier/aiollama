package sdcpp

// SDType mirrors sd_type_t from stable-diffusion.h.
type SDType int32

const (
	SDTypeF32 SDType = iota
	SDTypeF16
	SDTypeQ4_0
	SDTypeQ4_1
	SDTypeQ5_0
	SDTypeQ5_1
	SDTypeQ8_0
	SDTypeQ8_1
	SDTypeQ2_K
	SDTypeQ3_K
	SDTypeQ4_K
	SDTypeQ5_K
	SDTypeQ6_K
	SDTypeQ8_K
	SDTypeIQ2_XXS
	SDTypeIQ2_XS
	SDTypeIQ3_XXS
	SDTypeIQ1_S
	SDTypeIQ4_NL
	SDTypeIQ3_S
	SDTypeIQ2_S
	SDTypeIQ4_XS
	SDTypeI8
	SDTypeI16
	SDTypeI32
	SDTypeI64
	SDTypeF64
	SDTypeBF16
)

// SampleMethod mirrors sd_sample_method_t.
type SampleMethod int32

const (
	SampleEulerA SampleMethod = iota
	SampleEuler
	SampleHeun
	SampleDPM2
	SampleDPMPP2M
	SampleDPMPP2MA
	SampleLCM
	SampleDPMPPSDE
	SampleDPMFast
	SampleDPMAdaptive
	SampleTCD
	SampleEulerATrailing
	SampleEulerTrailing
)

// Schedule mirrors sd_schedule_t.
type Schedule int32

const (
	ScheduleDefault Schedule = iota
	ScheduleDiscrete
	ScheduleKarras
	ScheduleExponential
	ScheduleAYS
	ScheduleGITS
	ScheduleBeta
	ScheduleSimple
	ScheduleUniform
)

// CancelMode mirrors sd_cancel_mode_t.
type CancelMode int32

const (
	CancelNone CancelMode = iota
	CancelImage
	CancelAll
)

// PreviewMode mirrors preview_t.
type PreviewMode int32

const (
	PreviewNative PreviewMode = iota
	PreviewAccurate
)

// Image mirrors sd_image_t: raw RGB pixel data.
type Image struct {
	Width   int
	Height  int
	Channel int
	Data    []byte
}

// Audio mirrors sd_audio_t (stub for future LTXAV audio support).
type Audio struct {
	SampleRate   uint32
	SampleCount  uint32
	ChannelCount uint16
	Data         []byte
}

// SampleParams mirrors sd_sample_params_t.
type SampleParams struct {
	SampleSteps                     int32
	ThreadCount                     int32
	CFGScale                        float32
	Guidance                        float32
	ClipSkip                        float32
	SampleMethod                    SampleMethod
	Schedule                        Schedule
	FlowShift                      float32
	OldRateCoef                     float32
	OmegaAtX0                       float32
	OmegaAtXt                       float32
	OmegaAtVt                       float32
	SmptAt                          int32
	SmptAtDynamicThresholdingMax    int32
	VaryAtX0                        float32
	VaryAtXt                        int32
	X0Weight                        float32
	XtWeight                        int32
	Eta                             float32
	DiscreteFlowShift               int32
	NegTauAtX0                      int32
	NegTauAtXt                      float32
	UseKarras                       bool
	UseBetaDyShift                  bool
	BetaDyShiftStrength             float32
}

// TilingParams mirrors sd_tiling_params_t.
type TilingParams struct {
	Enable         bool
	UpscaleFactor  int32
	Strength       float32
	Denoise        float32
	ScaleEmphasis  float32
}

// CacheParams mirrors sd_cache_params_t.
type CacheParams struct {
	NoUnload                       bool
	ModelOffload                   bool
	MoeBoundary                    float32
	VaceStrength                   float32
	UseCache                       bool
	KeepModelLoaded                bool
	NegEmbdMask                    bool
	UseGuidance                    bool
	GuidanceScale                  float32
	SeedFrameIdx                   float32
}

// CtxParams mirrors sd_ctx_params_t. Only the fields the bridge populates are
// exposed; the C struct carries many optional component paths.
type CtxParams struct {
	ModelPath              string
	ClipLPath              string
	ClipGPath              string
	ClipVisionPath         string
	T5XXLPath              string
	VaePath                string
	TaesdPath              string
	ControlNetPath         string
	Backend                string
	ParamsBackend          string
	FlashAttn              bool
	DiffusionFlashAttn     bool
	VaeConvDirect          bool
	DiffusionConvDirect    bool
	WType                  SDType
	NThreads               int32
	EnableMmap             bool
	MaxVRAM                string
	StreamLayers           bool
}

// ImageGenParams mirrors sd_img_gen_params_t.
type ImageGenParams struct {
	Prompt            string
	NegativePrompt    string
	Width             int32
	Height            int32
	SampleParams      SampleParams
	Seed              int64
	BatchCount        int32
	InitImage         *Image
	MaskImage         *Image
	ControlImage      *Image
	ControlStrength   float32
	VAETilingParams   TilingParams
	IsKontext         bool
	Image2ImageStrength float32
	Image2ImageSteps    float32
}

// VideoGenParams mirrors sd_vid_gen_params_t.
type VideoGenParams struct {
	Prompt                  string
	NegativePrompt          string
	InitImage               *Image
	EndImage                *Image
	ControlFrames           []Image
	Width                   int32
	Height                  int32
	SampleParams            SampleParams
	HighNoiseSampleParams   SampleParams
	MoeBoundary             float32
	VaceStrength            float32
	Seed                    int64
	VideoFrames             int32
	FPS                     int32
	VAETilingParams         TilingParams
	Cache                   CacheParams
}

// ProgressFunc is the Go-side progress callback signature.
type ProgressFunc func(step, steps int, seconds float32)

// PreviewFunc is the Go-side preview callback signature.
type PreviewFunc func(step, frameCount int, frames []Image, isNoisy bool)

// LogFunc is the Go-side log callback signature.
type LogFunc func(level int32, text string)
