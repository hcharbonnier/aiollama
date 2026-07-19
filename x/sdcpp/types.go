package sdcpp

// SDType mirrors enum sd_type_t from stable-diffusion.h.
type SDType int32

const (
	SDTypeF32 SDType = 0
	SDTypeF16 SDType = 1

	SDTypeQ4_0 SDType = 2
	SDTypeQ4_1 SDType = 3

	SDTypeQ5_0  SDType = 6
	SDTypeQ5_1  SDType = 7
	SDTypeQ8_0  SDType = 8
	SDTypeQ8_1  SDType = 9
	SDTypeQ2_K  SDType = 10
	SDTypeQ3_K  SDType = 11
	SDTypeQ4_K  SDType = 12
	SDTypeQ5_K  SDType = 13
	SDTypeQ6_K  SDType = 14
	SDTypeQ8_K  SDType = 15
	SDTypeIQ2XXS SDType = 16
	SDTypeIQ2XS  SDType = 17
	SDTypeIQ3XXS SDType = 18
	SDTypeIQ1S   SDType = 19
	SDTypeIQ4NL  SDType = 20
	SDTypeIQ3S   SDType = 21
	SDTypeIQ2S   SDType = 22
	SDTypeIQ4XS  SDType = 23
	SDTypeI8     SDType = 24
	SDTypeI16    SDType = 25
	SDTypeI32    SDType = 26
	SDTypeI64    SDType = 27
	SDTypeF64    SDType = 28
	SDTypeIQ1M   SDType = 29
	SDTypeBF16   SDType = 30
	SDTypeTQ1_0  SDType = 34
	SDTypeTQ2_0  SDType = 35
	SDTypeMXFP4  SDType = 39
	SDTypeNVFP4  SDType = 40
	SDTypeQ1_0   SDType = 41
	SDTypeCount  SDType = 42
)

// SampleMethod mirrors enum sample_method_t.
type SampleMethod int32

const (
	SampleEuler SampleMethod = iota
	SampleEulerA
	SampleHeun
	SampleDPM2
	SampleDPMPP2SA
	SampleDPMPP2M
	SampleDPMPP2Mv2
	SampleIPNDM
	SampleIPNDMv
	SampleLCM
	SampleDDIMTrailing
	SampleTCD
	SampleResMultistep
	SampleRes2S
	SampleERSDE
	SampleEulerCfgPP
	SampleEulerACfgPP
	SampleEulerGE
	SampleDPMPP2MSDE
	SampleDPMPP2MSDEBT
)

// Schedule mirrors enum scheduler_t.
type Schedule int32

const (
	ScheduleDiscrete Schedule = iota
	ScheduleKarras
	ScheduleExponential
	ScheduleAYS
	ScheduleGITS
	ScheduleSGMUniform
	ScheduleSimple
	ScheduleSmoothstep
	ScheduleKLOptimal
	ScheduleLCM
	ScheduleBongTangent
	ScheduleLTX2
	ScheduleLogitNormal
	ScheduleFlux2
	ScheduleFlux
	ScheduleBeta
)

// Prediction mirrors enum prediction_t.
type Prediction int32

const (
	PredictionEPS Prediction = iota
	PredictionV
	PredictionEDMV
	PredictionFlow
	PredictionFluxFlow
	PredictionSEFIFlow
	PredictionMinit2IFlow
)

// CancelMode mirrors enum sd_cancel_mode_t.
type CancelMode int32

const (
	CancelAll CancelMode = iota
	CancelNewLatents
	CancelReset
)

// PreviewMode mirrors enum preview_t.
type PreviewMode int32

const (
	PreviewNone PreviewMode = iota
	PreviewProj
	PreviewTAE
	PreviewVAE
)

// RNGType mirrors enum rng_type_t.
type RNGType int32

const (
	RNGDefault RNGType = iota
	RNGCUDA
	RNGCPU
)

// LoraApplyMode mirrors enum lora_apply_mode_t.
type LoraApplyMode int32

const (
	LoraApplyAuto LoraApplyMode = iota
	LoraApplyImmediately
	LoraApplyAtRuntime
)

// VAEFormat mirrors enum sd_vae_format_t.
type VAEFormat int32

const (
	VAEFormatAuto  VAEFormat = -1
	VAEFormatFlux  VAEFormat = 0
	VAEFormatSD3   VAEFormat = 1
	VAEFormatFlux2 VAEFormat = 2
	VAEFormatWAN   VAEFormat = 3
)

// CacheMode mirrors enum sd_cache_mode_t.
type CacheMode int32

const (
	CacheDisabled CacheMode = iota
	CacheEasyCache
	CacheUCache
	CacheDBCache
	CacheTaylorSeer
	CacheCacheDit
	CacheSpectrum
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
	SampleRate  uint32
	Channels    uint32
	SampleCount uint64
	Data        []byte
}

// SLGParams mirrors sd_slg_params_t (skip-layer guidance).
type SLGParams struct {
	Layers     []int32
	LayerStart float32
	LayerEnd   float32
	Scale      float32
}

// GuidanceParams mirrors sd_guidance_params_t.
type GuidanceParams struct {
	TxtCfg            float32
	ImgCfg            float32
	DistilledGuidance float32
	SLG               SLGParams
}

// SampleParams mirrors sd_sample_params_t.
type SampleParams struct {
	Guidance           GuidanceParams
	Schedule           Schedule
	SampleMethod       SampleMethod
	SampleSteps        int32
	Eta                float32
	ShiftedTimestep    int32
	CustomSigmas       []float32
	FlowShift          float32
	ExtraSampleArgs    string
}

// TilingParams mirrors sd_tiling_params_t.
type TilingParams struct {
	Enabled         bool
	TemporalTiling  bool
	TileSizeX       int32
	TileSizeY       int32
	TargetOverlap   float32
	RelSizeX        float32
	RelSizeY        float32
	ExtraTilingArgs string
}

// PMParams mirrors sd_pm_params_t (photo maker).
type PMParams struct {
	IDImages      []Image
	IDEmbedPath   string
	StyleStrength  float32
}

// PulidParams mirrors sd_pulid_params_t.
type PulidParams struct {
	IDEmbeddingPath string
	IDWeight        float32
}

// CacheParams mirrors sd_cache_params_t.
type CacheParams struct {
	Mode                       CacheMode
	ReuseThreshold             float32
	StartPercent               float32
	EndPercent                 float32
	ErrorDecayRate             float32
	UseRelativeThreshold       bool
	ResetErrorOnCompute        bool
	FnComputeBlocks            int32
	BnComputeBlocks            int32
	ResidualDiffThreshold      float32
	MaxWarmupSteps             int32
	MaxCachedSteps             int32
	MaxContinuousCachedSteps    int32
	TaylorseerNDerivatives     int32
	TaylorseerSkipInterval     int32
	SCMMask                    string
	SCMPolicyDynamic           bool
	SpectrumW                  float32
	SpectrumM                  int32
	SpectrumLam                float32
	SpectrumWindowSize         int32
	SpectrumFlexWindow         float32
	SpectrumWarmupSteps        int32
	SpectrumStopPercent        float32
}

// Lora mirrors sd_lora_t.
type Lora struct {
	IsHighNoise bool
	Multiplier  float32
	Path        string
}

// HiresUpsscaler mirrors enum sd_hires_upscaler_t.
type HiresUpscaler int32

const (
	HiresUpscalerNone HiresUpscaler = iota
	HiresUpscalerLatent
	HiresUpscalerLatentNearest
	HiresUpscalerLatentNearestExact
	HiresUpscalerLatentAntialiased
	HiresUpscalerLatentBicubic
	HiresUpscalerLatentBicubicAntialiased
	HiresUpscalerLanczos
	HiresUpscalerNearest
	HiresUpscalerModel
)

// HiresParams mirrors sd_hires_params_t.
type HiresParams struct {
	Enabled             bool
	Upscaler            HiresUpscaler
	ModelPath           string
	Scale               float32
	TargetWidth         int32
	TargetHeight        int32
	Steps               int32
	DenoisingStrength   float32
	UpscaleTileSize     int32
	CustomSigmas        []float32
}

// CtxParams mirrors sd_ctx_params_t. Only the fields the bridge populates are
// exposed; the C struct carries many optional component paths.
type CtxParams struct {
	ModelPath                     string
	ClipLPath                     string
	ClipGPath                     string
	ClipVisionPath                string
	T5XXLPath                     string
	LLMPath                       string
	LLMVisionPath                 string
	DiffusionModelPath            string
	HighNoiseDiffusionModelPath   string
	UncondDiffusionModelPath      string
	EmbeddingsConnectorsPath     string
	VaePath                       string
	AudioVaePath                  string
	TaesdPath                     string
	ControlNetPath                string
	MotionModulePath              string
	PhotoMakerPath                string
	PulidWeightsPath              string
	TensorTypeRules               string
	NThreads                      int32
	WType                         SDType
	RNGType                       RNGType
	SamplerRNGType                RNGType
	Prediction                    Prediction
	LoraApplyMode                 LoraApplyMode
	EnableMmap                    bool
	FlashAttn                     bool
	DiffusionFlashAttn            bool
	TaePreviewOnly                bool
	DiffusionConvDirect           bool
	VaeConvDirect                 bool
	ForceSDXLVAEConvScale         bool
	VAEFormat                     VAEFormat
	MaxVRAM                       string
	StreamLayers                  bool
	EagerLoad                     bool
	Backend                       string
	ParamsBackend                 string
	SplitMode                     string
	AutoFit                       bool
	RPCServers                    string
	ModelArgs                     string
}

// ImageGenParams mirrors sd_img_gen_params_t.
type ImageGenParams struct {
	Loras            []Lora
	Prompt           string
	NegativePrompt   string
	ClipSkip         int32
	InitImage        *Image
	RefImages        []Image
	MaskImage        *Image
	Width            int32
	Height           int32
	SampleParams     SampleParams
	Strength         float32
	Seed             int64
	BatchCount       int32
	ControlImage     *Image
	ControlStrength  float32
	PMParams         PMParams
	PulidParams      PulidParams
	VAETilingParams  TilingParams
	Cache            CacheParams
	Hires            HiresParams
	QwenImageLayers  int32
	CircularX        bool
	CircularY        bool
}

// VideoGenParams mirrors sd_vid_gen_params_t.
type VideoGenParams struct {
	Loras                  []Lora
	Prompt                 string
	NegativePrompt         string
	ClipSkip               int32
	InitImage              *Image
	EndImage               *Image
	ControlFrames          []Image
	Width                  int32
	Height                 int32
	SampleParams           SampleParams
	HighNoiseSampleParams  SampleParams
	MoeBoundary            float32
	Strength               float32
	Seed                   int64
	VideoFrames            int32
	FPS                    int32
	VaceStrength           float32
	VAETilingParams        TilingParams
	Cache                  CacheParams
	Hires                  HiresParams
	CircularX              bool
	CircularY              bool
}

// ProgressFunc is the Go-side progress callback signature.
type ProgressFunc func(step, steps int, seconds float32)

// PreviewFunc is the Go-side preview callback signature.
type PreviewFunc func(step, frameCount int, frames []Image, isNoisy bool)

// LogFunc is the Go-side log callback signature.
type LogFunc func(level int32, text string)
