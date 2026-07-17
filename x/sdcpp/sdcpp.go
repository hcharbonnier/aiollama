// Package sdcpp provides a CGO bridge to the stable-diffusion.cpp C library,
// replacing the former MLX bridge (x/imagegen/mlx). It wraps image and video
// generation, progress/preview/log callbacks, cancellation, and capability
// queries.
//
// The bridge links against libstable-diffusion built by cmake/sdcpp. Callbacks
// are trampolined from C into Go via cgo.Handle, so the C ABI never calls back
// into a moving Go closure directly.
package sdcpp

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/include
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++
#include "stable-diffusion.h"
#include <stdlib.h>
#include <string.h>

// Trampolines that forward C callbacks into the Go registry (exported below).
extern void goProgressTrampoline(int step, int steps, float time, void* data);
extern void goPreviewTrampoline(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);
extern void goLogTrampoline(enum sd_log_level_t level, char* text, void* data);

// goLogTrampolineConst is a thin C wrapper that adapts the const-qualified
// sd_log_cb_t signature to the non-const goLogTrampoline export (cgo does not
// generate const-qualified pointer exports). Passing goLogTrampoline directly
// triggers -Wincompatible-pointer-types because the typedef expects
// const char*.
static void goLogTrampolineConst(enum sd_log_level_t level, const char* text, void* data) {
    goLogTrampoline(level, (char*)text, data);
}

// sd_install_progress installs the progress + preview trampolines with the
// given data value (a cgo.Handle uintptr). The uintptr->void* cast happens
// here in C to avoid a Go-side unsafe.Pointer(uintptr) conversion that vet
// flags as a possible misuse. Called per-generate to route callbacks to the
// active request.
static void sd_install_progress(uintptr_t data) {
    void* p = (void*)data;
    sd_set_progress_callback(goProgressTrampoline, p);
    sd_set_preview_callback(goPreviewTrampoline, PREVIEW_NONE, 1, false, false, p);
}

// sd_clear_progress uninstalls the progress + preview trampolines.
static void sd_clear_progress(void) {
    sd_set_progress_callback(NULL, NULL);
    sd_set_preview_callback(NULL, PREVIEW_NONE, 1, false, false, NULL);
}

// sd_install_log installs the global log trampoline.
static void sd_install_log(void) {
    sd_set_log_callback(goLogTrampolineConst, NULL);
}
*/
import "C"

import (
	"fmt"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// Context is an opaque handle to an SD.cpp context (sd_ctx_t*).
type Context struct {
	handle *C.sd_ctx_t
}

// callbacksReady ensures the C trampolines are installed once per process.
// SD.cpp stores a single process-global callback slot, so the active request
// installs its own cgo.Handle as the data pointer before each generate call
// and removes it afterward. The diffGenMu mutex in the runner serializes
// generation so only one callback is active at a time.
var callbacksReady sync.Once

// ensureCallbacks installs the C trampolines once.
func ensureCallbacks() {
	callbacksReady.Do(func() {
		C.sd_install_log()
	})
}

//export goProgressTrampoline
func goProgressTrampoline(step, steps C.int, time C.float, data unsafe.Pointer) {
	if data == nil {
		return
	}
	h := cgo.Handle(data).Value()
	if h == nil {
		return
	}
	fn, ok := h.(ProgressFunc)
	if !ok {
		return
	}
	fn(int(step), int(steps), float32(time))
}

//export goPreviewTrampoline
func goPreviewTrampoline(step, frameCount C.int, frames *C.sd_image_t, isNoisy C.bool, data unsafe.Pointer) {
	if data == nil {
		return
	}
	h := cgo.Handle(data).Value()
	if h == nil {
		return
	}
	fn, ok := h.(PreviewFunc)
	if !ok {
		return
	}
	var imgs []Image
	if frames != nil && frameCount > 0 {
		slice := (*[1 << 20]C.sd_image_t)(unsafe.Pointer(frames))[:frameCount:frameCount]
		for i := range slice {
			imgs = append(imgs, cImageToGo(&slice[i]))
		}
	}
	fn(int(step), int(frameCount), imgs, bool(isNoisy))
}

//export goLogTrampoline
func goLogTrampoline(level C.enum_sd_log_level_t, text *C.char, data unsafe.Pointer) {
	if text == nil {
		return
	}
	msg := C.GoString(text)
	if globalLogSink != nil {
		globalLogSink(int32(level), msg)
	}
}

// globalLogSink is the process-wide log sink, set via SetLogCallback.
var globalLogSink LogFunc

// SetLogCallback registers a process-wide log sink. Pass nil to clear.
func SetLogCallback(fn LogFunc) {
	globalLogSink = fn
}

// installProgressCallback registers the given progress callback for the
// duration of a generate call by installing the C trampolines with a
// cgo.Handle as the data pointer. Returns the handle, which the caller must
// pass to clearProgressCallback after generation completes.
func installProgressCallback(fn ProgressFunc) cgo.Handle {
	if fn == nil {
		C.sd_clear_progress()
		return 0
	}
	handle := cgo.NewHandle(fn)
	C.sd_install_progress(C.uintptr_t(handle))
	return handle
}

// clearProgressCallback uninstalls the trampolines and deletes the handle.
func clearProgressCallback(handle cgo.Handle) {
	if handle == 0 {
		return
	}
	C.sd_clear_progress()
	handle.Delete()
}

// cstr allocates a C string from a Go string and returns the pointer plus a
// cleanup closure. The deferred free must be called by the caller.
func cstr(s string) (*C.char, func()) {
	if s == "" {
		return nil, func() {}
	}
	cs := C.CString(s)
	return cs, func() { C.free(unsafe.Pointer(cs)) }
}

// NewContext creates an SD.cpp context from the given params. The caller must
// Close it when done.
func NewContext(p CtxParams) (*Context, error) {
	ensureCallbacks()

	// sd_ctx_params_init zero-initializes the struct and sets default values
	// for fields the bridge does not populate. This is critical now that the
	// struct carries enums (rng_type, prediction, lora_apply_mode, vae_format)
	// and pointer fields that must start at their default/zero values rather
	// than whatever stack garbage an uninitialized C.sd_ctx_params_t{} would
	// contain.
	cp := C.sd_ctx_params_t{}
	C.sd_ctx_params_init(&cp)

	model, freeModel := cstr(p.ModelPath)
	defer freeModel()
	cp.model_path = model

	clipL, freeClipL := cstr(p.ClipLPath)
	defer freeClipL()
	cp.clip_l_path = clipL

	clipG, freeClipG := cstr(p.ClipGPath)
	defer freeClipG()
	cp.clip_g_path = clipG

	clipVision, freeClipVision := cstr(p.ClipVisionPath)
	defer freeClipVision()
	cp.clip_vision_path = clipVision

	t5xxl, freeT5xxl := cstr(p.T5XXLPath)
	defer freeT5xxl()
	cp.t5xxl_path = t5xxl

	llm, freeLLM := cstr(p.LLMPath)
	defer freeLLM()
	cp.llm_path = llm

	llmVision, freeLLMVision := cstr(p.LLMVisionPath)
	defer freeLLMVision()
	cp.llm_vision_path = llmVision

	diffModel, freeDiffModel := cstr(p.DiffusionModelPath)
	defer freeDiffModel()
	cp.diffusion_model_path = diffModel

	highNoise, freeHighNoise := cstr(p.HighNoiseDiffusionModelPath)
	defer freeHighNoise()
	cp.high_noise_diffusion_model_path = highNoise

	uncond, freeUncond := cstr(p.UncondDiffusionModelPath)
	defer freeUncond()
	cp.uncond_diffusion_model_path = uncond

	embConn, freeEmbConn := cstr(p.EmbeddingsConnectorsPath)
	defer freeEmbConn()
	cp.embeddings_connectors_path = embConn

	vae, freeVae := cstr(p.VaePath)
	defer freeVae()
	cp.vae_path = vae

	audioVae, freeAudioVae := cstr(p.AudioVaePath)
	defer freeAudioVae()
	cp.audio_vae_path = audioVae

	taesd, freeTaesd := cstr(p.TaesdPath)
	defer freeTaesd()
	cp.taesd_path = taesd

	controlNet, freeControlNet := cstr(p.ControlNetPath)
	defer freeControlNet()
	cp.control_net_path = controlNet

	motion, freeMotion := cstr(p.MotionModulePath)
	defer freeMotion()
	cp.motion_module_path = motion

	photoMaker, freePhotoMaker := cstr(p.PhotoMakerPath)
	defer freePhotoMaker()
	cp.photo_maker_path = photoMaker

	pulid, freePulid := cstr(p.PulidWeightsPath)
	defer freePulid()
	cp.pulid_weights_path = pulid

	tensorRules, freeTensorRules := cstr(p.TensorTypeRules)
	defer freeTensorRules()
	cp.tensor_type_rules = tensorRules

	backend, freeBackend := cstr(p.Backend)
	defer freeBackend()
	cp.backend = backend

	paramsBackend, freeParamsBackend := cstr(p.ParamsBackend)
	defer freeParamsBackend()
	cp.params_backend = paramsBackend

	splitMode, freeSplitMode := cstr(p.SplitMode)
	defer freeSplitMode()
	cp.split_mode = splitMode

	maxVRAM, freeMaxVRAM := cstr(p.MaxVRAM)
	defer freeMaxVRAM()
	cp.max_vram = maxVRAM

	rpcServers, freeRPC := cstr(p.RPCServers)
	defer freeRPC()
	cp.rpc_servers = rpcServers

	modelArgs, freeModelArgs := cstr(p.ModelArgs)
	defer freeModelArgs()
	cp.model_args = modelArgs

	cp.n_threads = C.int(p.NThreads)
	cp.wtype = C.enum_sd_type_t(p.WType)
	cp.rng_type = C.enum_rng_type_t(p.RNGType)
	cp.sampler_rng_type = C.enum_rng_type_t(p.SamplerRNGType)
	cp.prediction = C.enum_prediction_t(p.Prediction)
	cp.lora_apply_mode = C.enum_lora_apply_mode_t(p.LoraApplyMode)
	cp.enable_mmap = C.bool(p.EnableMmap)
	cp.flash_attn = C.bool(p.FlashAttn)
	cp.diffusion_flash_attn = C.bool(p.DiffusionFlashAttn)
	cp.tae_preview_only = C.bool(p.TaePreviewOnly)
	cp.diffusion_conv_direct = C.bool(p.DiffusionConvDirect)
	cp.vae_conv_direct = C.bool(p.VaeConvDirect)
	cp.force_sdxl_vae_conv_scale = C.bool(p.ForceSDXLVAEConvScale)
	cp.vae_format = C.enum_sd_vae_format_t(p.VAEFormat)
	cp.stream_layers = C.bool(p.StreamLayers)
	cp.eager_load = C.bool(p.EagerLoad)
	cp.auto_fit = C.bool(p.AutoFit)

	ctx := C.new_sd_ctx(&cp)
	if ctx == nil {
		return nil, fmt.Errorf("new_sd_ctx returned null (check model paths and backend availability)")
	}
	return &Context{handle: ctx}, nil
}

// Close frees the underlying SD.cpp context.
func (c *Context) Close() {
	if c != nil && c.handle != nil {
		C.free_sd_ctx(c.handle)
		c.handle = nil
	}
}

// SupportsImageGeneration reports whether the loaded context can generate images.
func (c *Context) SupportsImageGeneration() bool {
	if c == nil || c.handle == nil {
		return false
	}
	return bool(C.sd_ctx_supports_image_generation(c.handle))
}

// SupportsVideoGeneration reports whether the loaded context can generate video.
func (c *Context) SupportsVideoGeneration() bool {
	if c == nil || c.handle == nil {
		return false
	}
	return bool(C.sd_ctx_supports_video_generation(c.handle))
}

// CancelGeneration requests cancellation of in-progress generation.
func (c *Context) CancelGeneration(mode CancelMode) {
	if c == nil || c.handle == nil {
		return
	}
	C.sd_cancel_generation(c.handle, C.enum_sd_cancel_mode_t(mode))
}

// GenerateImage runs image generation synchronously and returns the produced
// images. The caller does not free the returned data; it is copied into Go
// slices before the C buffers are freed.
func (c *Context) GenerateImage(p ImageGenParams, progress ProgressFunc) ([]Image, error) {
	if c == nil || c.handle == nil {
		return nil, fmt.Errorf("nil context")
	}

	// sd_img_gen_params_init zero-initializes + sets defaults for the many
	// pointer/enum fields the bridge does not populate.
	params := C.sd_img_gen_params_t{}
	C.sd_img_gen_params_init(&params)

	prompt, freePrompt := cstr(p.Prompt)
	defer freePrompt()
	params.prompt = prompt

	negPrompt, freeNeg := cstr(p.NegativePrompt)
	defer freeNeg()
	params.negative_prompt = negPrompt

	params.clip_skip = C.int(p.ClipSkip)

	if p.InitImage != nil {
		params.init_image = goImageToC(p.InitImage)
		defer C.free(unsafe.Pointer(params.init_image.data))
	}
	if p.MaskImage != nil {
		params.mask_image = goImageToC(p.MaskImage)
		defer C.free(unsafe.Pointer(params.mask_image.data))
	}
	if p.ControlImage != nil {
		params.control_image = goImageToC(p.ControlImage)
		defer C.free(unsafe.Pointer(params.control_image.data))
	}
	params.width = C.int(p.Width)
	params.height = C.int(p.Height)
	params.sample_params = goSampleParamsToC(p.SampleParams)
	params.strength = C.float(p.Strength)
	params.seed = C.int64_t(p.Seed)
	params.batch_count = C.int(p.BatchCount)
	params.control_strength = C.float(p.ControlStrength)
	params.vae_tiling_params = goTilingParamsToC(p.VAETilingParams)
	params.cache = goCacheParamsToC(p.Cache)
	params.hires = goHiresParamsToC(p.Hires)
	params.qwen_image_layers = C.int(p.QwenImageLayers)
	params.circular_x = C.bool(p.CircularX)
	params.circular_y = C.bool(p.CircularY)

	progHandle := installProgressCallback(progress)
	defer clearProgressCallback(progHandle)

	var imagesOut *C.sd_image_t
	var numOut C.int
	ok := C.generate_image(c.handle, &params, &imagesOut, &numOut)
	if imagesOut != nil {
		defer C.free_sd_images(imagesOut, numOut)
	}
	if !ok {
		return nil, fmt.Errorf("generate_image failed")
	}

	n := int(numOut)
	if n <= 0 || imagesOut == nil {
		return nil, fmt.Errorf("generate_image produced no images")
	}
	slice := (*[1 << 20]C.sd_image_t)(unsafe.Pointer(imagesOut))[:n:n]
	out := make([]Image, 0, n)
	for i := range slice {
		out = append(out, cImageToGo(&slice[i]))
	}
	return out, nil
}

// GenerateVideo runs video generation synchronously and returns the produced
// frames (and optional audio, currently nil).
func (c *Context) GenerateVideo(p VideoGenParams, progress ProgressFunc) ([]Image, error) {
	if c == nil || c.handle == nil {
		return nil, fmt.Errorf("nil context")
	}

	params := C.sd_vid_gen_params_t{}
	C.sd_vid_gen_params_init(&params)

	prompt, freePrompt := cstr(p.Prompt)
	defer freePrompt()
	params.prompt = prompt

	negPrompt, freeNeg := cstr(p.NegativePrompt)
	defer freeNeg()
	params.negative_prompt = negPrompt

	params.clip_skip = C.int(p.ClipSkip)

	if p.InitImage != nil {
		params.init_image = goImageToC(p.InitImage)
		defer C.free(unsafe.Pointer(params.init_image.data))
	}
	if p.EndImage != nil {
		params.end_image = goImageToC(p.EndImage)
		defer C.free(unsafe.Pointer(params.end_image.data))
	}
	params.width = C.int(p.Width)
	params.height = C.int(p.Height)
	params.sample_params = goSampleParamsToC(p.SampleParams)
	params.high_noise_sample_params = goSampleParamsToC(p.HighNoiseSampleParams)
	params.moe_boundary = C.float(p.MoeBoundary)
	params.strength = C.float(p.Strength)
	params.seed = C.int64_t(p.Seed)
	params.video_frames = C.int(p.VideoFrames)
	params.fps = C.int(p.FPS)
	params.vace_strength = C.float(p.VaceStrength)
	params.vae_tiling_params = goTilingParamsToC(p.VAETilingParams)
	params.cache = goCacheParamsToC(p.Cache)
	params.hires = goHiresParamsToC(p.Hires)
	params.circular_x = C.bool(p.CircularX)
	params.circular_y = C.bool(p.CircularY)

	progHandle := installProgressCallback(progress)
	defer clearProgressCallback(progHandle)

	var framesOut *C.sd_image_t
	var numFrames C.int
	var audioOut *C.sd_audio_t
	ok := C.generate_video(c.handle, &params, &framesOut, &numFrames, &audioOut)
	if framesOut != nil {
		defer C.free_sd_images(framesOut, numFrames)
	}
	if audioOut != nil {
		defer C.free_sd_audio(audioOut)
	}
	if !ok {
		return nil, fmt.Errorf("generate_video failed")
	}

	n := int(numFrames)
	if n <= 0 || framesOut == nil {
		return nil, fmt.Errorf("generate_video produced no frames")
	}
	slice := (*[1 << 20]C.sd_image_t)(unsafe.Pointer(framesOut))[:n:n]
	out := make([]Image, 0, n)
	for i := range slice {
		out = append(out, cImageToGo(&slice[i]))
	}
	return out, nil
}

// SystemInfo returns the SD.cpp system/device info string.
func SystemInfo() string {
	s := C.sd_get_system_info()
	if s == nil {
		return ""
	}
	defer C.free(unsafe.Pointer(s))
	return C.GoString(s)
}

// ListDevices returns the list of available ggml backend devices as a string,
// one "name<TAB>description" per line. Returns an empty string if no devices
// are available or the query fails.
func ListDevices() string {
	// First query the required size.
	required := C.sd_list_devices(nil, 0)
	if required == 0 {
		return ""
	}
	buf := make([]byte, int(required)+1)
	C.sd_list_devices((*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	// Trim at the first null byte in case the buffer is larger than needed.
	for i, b := range buf {
		if b == 0 {
			return string(buf[:i])
		}
	}
	return string(buf)
}

// Commit returns the SD.cpp git commit hash.
func Commit() string {
	s := C.sd_commit()
	if s == nil {
		return ""
	}
	return C.GoString(s)
}

// Version returns the SD.cpp version string.
func Version() string {
	s := C.sd_version()
	if s == nil {
		return ""
	}
	return C.GoString(s)
}

// Convert invokes SD.cpp's model format conversion (e.g. PyTorch -> safetensors/GGUF).
func Convert(inputPath, vaePath, outputPath string, outputType SDType, tensorTypeRules string, convertName bool) error {
	cIn, freeIn := cstr(inputPath)
	defer freeIn()
	cVae, freeVae := cstr(vaePath)
	defer freeVae()
	cOut, freeOut := cstr(outputPath)
	defer freeOut()
	cRules, freeRules := cstr(tensorTypeRules)
	defer freeRules()
	if ok := C.convert(cIn, cVae, cOut, C.enum_sd_type_t(outputType), cRules, C.bool(convertName)); !ok {
		return fmt.Errorf("convert failed for %s", inputPath)
	}
	return nil
}

func goImageToC(img *Image) C.sd_image_t {
	if img == nil {
		return C.sd_image_t{}
	}
	var dataPtr *C.uint8_t
	if len(img.Data) > 0 {
		dataPtr = (*C.uint8_t)(C.CBytes(img.Data))
	}
	return C.sd_image_t{
		width:   C.uint32_t(img.Width),
		height:  C.uint32_t(img.Height),
		channel: C.uint32_t(img.Channel),
		data:    dataPtr,
	}
}

func cImageToGo(img *C.sd_image_t) Image {
	out := Image{
		Width:   int(img.width),
		Height:  int(img.height),
		Channel: int(img.channel),
	}
	if img.data != nil {
		size := out.Width * out.Height * out.Channel
		if size > 0 {
			out.Data = C.GoBytes(unsafe.Pointer(img.data), C.int(size))
		}
	}
	return out
}

func goSampleParamsToC(p SampleParams) C.sd_sample_params_t {
	sp := C.sd_sample_params_t{}
	C.sd_sample_params_init(&sp)
	sp.guidance = goGuidanceParamsToC(p.Guidance)
	sp.scheduler = C.enum_scheduler_t(p.Schedule)
	sp.sample_method = C.enum_sample_method_t(p.SampleMethod)
	sp.sample_steps = C.int(p.SampleSteps)
	sp.eta = C.float(p.Eta)
	sp.shifted_timestep = C.int(p.ShiftedTimestep)
	sp.flow_shift = C.float(p.FlowShift)
	if p.ExtraSampleArgs != "" {
		sp.extra_sample_args = C.CString(p.ExtraSampleArgs)
		// Note: leaked if non-empty; sample params are short-lived stack
		// structs passed by value to a synchronous call. The strdup is
		// intentional to avoid a use-after-free if the C side retains the
		// pointer past the call. In practice the bridge never sets this.
	}
	return sp
}

func goGuidanceParamsToC(p GuidanceParams) C.sd_guidance_params_t {
	return C.sd_guidance_params_t{
		txt_cfg:            C.float(p.TxtCfg),
		img_cfg:            C.float(p.ImgCfg),
		distilled_guidance: C.float(p.DistilledGuidance),
		slg: C.sd_slg_params_t{
			layer_start: C.float(p.SLG.LayerStart),
			layer_end:   C.float(p.SLG.LayerEnd),
			scale:       C.float(p.SLG.Scale),
		},
	}
}

func goTilingParamsToC(p TilingParams) C.sd_tiling_params_t {
	tp := C.sd_tiling_params_t{}
	tp.enabled = C.bool(p.Enabled)
	tp.temporal_tiling = C.bool(p.TemporalTiling)
	tp.tile_size_x = C.int(p.TileSizeX)
	tp.tile_size_y = C.int(p.TileSizeY)
	tp.target_overlap = C.float(p.TargetOverlap)
	tp.rel_size_x = C.float(p.RelSizeX)
	tp.rel_size_y = C.float(p.RelSizeY)
	if p.ExtraTilingArgs != "" {
		tp.extra_tiling_args = C.CString(p.ExtraTilingArgs)
	}
	return tp
}

func goCacheParamsToC(p CacheParams) C.sd_cache_params_t {
	cp := C.sd_cache_params_t{}
	C.sd_cache_params_init(&cp)
	cp.mode = C.enum_sd_cache_mode_t(p.Mode)
	cp.reuse_threshold = C.float(p.ReuseThreshold)
	cp.start_percent = C.float(p.StartPercent)
	cp.end_percent = C.float(p.EndPercent)
	cp.error_decay_rate = C.float(p.ErrorDecayRate)
	cp.use_relative_threshold = C.bool(p.UseRelativeThreshold)
	cp.reset_error_on_compute = C.bool(p.ResetErrorOnCompute)
	cp.Fn_compute_blocks = C.int(p.FnComputeBlocks)
	cp.Bn_compute_blocks = C.int(p.BnComputeBlocks)
	cp.residual_diff_threshold = C.float(p.ResidualDiffThreshold)
	cp.max_warmup_steps = C.int(p.MaxWarmupSteps)
	cp.max_cached_steps = C.int(p.MaxCachedSteps)
	cp.max_continuous_cached_steps = C.int(p.MaxContinuousCachedSteps)
	cp.taylorseer_n_derivatives = C.int(p.TaylorseerNDerivatives)
	cp.taylorseer_skip_interval = C.int(p.TaylorseerSkipInterval)
	cp.scm_policy_dynamic = C.bool(p.SCMPolicyDynamic)
	cp.spectrum_w = C.float(p.SpectrumW)
	cp.spectrum_m = C.int(p.SpectrumM)
	cp.spectrum_lam = C.float(p.SpectrumLam)
	cp.spectrum_window_size = C.int(p.SpectrumWindowSize)
	cp.spectrum_flex_window = C.float(p.SpectrumFlexWindow)
	cp.spectrum_warmup_steps = C.int(p.SpectrumWarmupSteps)
	cp.spectrum_stop_percent = C.float(p.SpectrumStopPercent)
	return cp
}

func goHiresParamsToC(p HiresParams) C.sd_hires_params_t {
	hp := C.sd_hires_params_t{}
	C.sd_hires_params_init(&hp)
	hp.enabled = C.bool(p.Enabled)
	hp.upscaler = C.enum_sd_hires_upscaler_t(p.Upscaler)
	if p.ModelPath != "" {
		hp.model_path = C.CString(p.ModelPath)
	}
	hp.scale = C.float(p.Scale)
	hp.target_width = C.int(p.TargetWidth)
	hp.target_height = C.int(p.TargetHeight)
	hp.steps = C.int(p.Steps)
	hp.denoising_strength = C.float(p.DenoisingStrength)
	hp.upscale_tile_size = C.int(p.UpscaleTileSize)
	return hp
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
