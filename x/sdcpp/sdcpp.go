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
extern void goLogTrampoline(int level, char* text, void* data);

// sd_install_progress installs the progress + preview trampolines with the
// given data value (a cgo.Handle uintptr). The uintptr->void* cast happens
// here in C to avoid a Go-side unsafe.Pointer(uintptr) conversion that vet
// flags as a possible misuse. Called per-generate to route callbacks to the
// active request.
static void sd_install_progress(uintptr_t data) {
    void* p = (void*)data;
    sd_set_progress_callback(goProgressTrampoline, p);
    sd_set_preview_callback(goPreviewTrampoline, SD_PREVIEW_NATIVE, 1, false, false, p);
}

// sd_clear_progress uninstalls the progress + preview trampolines.
static void sd_clear_progress(void) {
    sd_set_progress_callback(NULL, NULL);
    sd_set_preview_callback(NULL, SD_PREVIEW_NATIVE, 1, false, false, NULL);
}

// sd_install_log installs the global log trampoline.
static void sd_install_log(void) {
    sd_set_log_callback(goLogTrampoline, NULL);
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
		// frames is a contiguous array of sd_image_t.
		slice := (*[1 << 20]C.sd_image_t)(unsafe.Pointer(frames))[:frameCount:frameCount]
		for i := range slice {
			imgs = append(imgs, cImageToGo(&slice[i]))
		}
	}
	fn(int(step), int(frameCount), imgs, bool(isNoisy))
}

//export goLogTrampoline
func goLogTrampoline(level C.int, text *C.char, data unsafe.Pointer) {
	if text == nil {
		return
	}
	// Log callback is process-global (no per-request handle); broadcast to
	// the package-level log sink. There is no per-request routing because
	// SD.cpp installs a single global log callback.
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
// pass to clearProgressCallback after generation completes. Since SD.cpp uses a
// single process-global callback slot, the caller must serialize generation
// (the runner does this via diffGenMu).
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
// Calling with handle 0 is a no-op.
func clearProgressCallback(handle cgo.Handle) {
	if handle == 0 {
		return
	}
	C.sd_clear_progress()
	handle.Delete()
}

// NewContext creates an SD.cpp context from the given params. The caller must
// Close it when done.
func NewContext(p CtxParams) (*Context, error) {
	ensureCallbacks()

	cp := C.sd_ctx_params_t{}
	if p.ModelPath != "" {
		cp.model_path = C.CString(p.ModelPath)
		defer C.free(unsafe.Pointer(cp.model_path))
	}
	if p.ClipLPath != "" {
		cp.clip_l_path = C.CString(p.ClipLPath)
		defer C.free(unsafe.Pointer(cp.clip_l_path))
	}
	if p.ClipGPath != "" {
		cp.clip_g_path = C.CString(p.ClipGPath)
		defer C.free(unsafe.Pointer(cp.clip_g_path))
	}
	if p.ClipVisionPath != "" {
		cp.clip_vision_path = C.CString(p.ClipVisionPath)
		defer C.free(unsafe.Pointer(cp.clip_vision_path))
	}
	if p.T5XXLPath != "" {
		cp.t5xxl_path = C.CString(p.T5XXLPath)
		defer C.free(unsafe.Pointer(cp.t5xxl_path))
	}
	if p.VaePath != "" {
		cp.vae_path = C.CString(p.VaePath)
		defer C.free(unsafe.Pointer(cp.vae_path))
	}
	if p.TaesdPath != "" {
		cp.taesd_path = C.CString(p.TaesdPath)
		defer C.free(unsafe.Pointer(cp.taesd_path))
	}
	if p.ControlNetPath != "" {
		cp.control_net_path = C.CString(p.ControlNetPath)
		defer C.free(unsafe.Pointer(cp.control_net_path))
	}
	if p.Backend != "" {
		cp.backend = C.CString(p.Backend)
		defer C.free(unsafe.Pointer(cp.backend))
	}
	if p.ParamsBackend != "" {
		cp.params_backend = C.CString(p.ParamsBackend)
		defer C.free(unsafe.Pointer(cp.params_backend))
	}
	if p.MaxVRAM != "" {
		cp.max_vram = C.CString(p.MaxVRAM)
		defer C.free(unsafe.Pointer(cp.max_vram))
	}
	cp.flash_attn = C.bool(p.FlashAttn)
	cp.diffusion_flash_attn = C.bool(p.DiffusionFlashAttn)
	cp.vae_conv_direct = C.bool(p.VaeConvDirect)
	cp.diffusion_conv_direct = C.bool(p.DiffusionConvDirect)
	cp.wtype = C.sd_type_t(p.WType)
	cp.n_threads = C.int(p.NThreads)
	cp.enable_mmap = C.bool(p.EnableMmap)
	cp.stream_layers = C.bool(p.StreamLayers)

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
	C.sd_cancel_generation(c.handle, C.sd_cancel_mode_t(mode))
}

// GenerateImage runs image generation synchronously and returns the produced
// images. The caller does not free the returned data; it is copied into Go
// slices before the C buffers are freed.
func (c *Context) GenerateImage(p ImageGenParams, progress ProgressFunc) ([]Image, error) {
	if c == nil || c.handle == nil {
		return nil, fmt.Errorf("nil context")
	}

	params := C.sd_img_gen_params_t{}
	if p.Prompt != "" {
		params.prompt = C.CString(p.Prompt)
		defer C.free(unsafe.Pointer(params.prompt))
	}
	if p.NegativePrompt != "" {
		params.negative_prompt = C.CString(p.NegativePrompt)
		defer C.free(unsafe.Pointer(params.negative_prompt))
	}
	params.width = C.int(p.Width)
	params.height = C.int(p.Height)
	params.sample_params = goSampleParamsToC(p.SampleParams)
	params.seed = C.int64_t(p.Seed)
	params.batch_count = C.int(p.BatchCount)
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
	params.control_strength = C.float(p.ControlStrength)
	params.vae_tiling_params = goTilingParamsToC(p.VAETilingParams)
	params.is_kontext = C.bool(p.IsKontext)
	params.image2image_strength = C.float(p.Image2ImageStrength)
	params.image2image_steps = C.float(p.Image2ImageSteps)

	progHandle := installProgressCallback(progress)
	defer clearProgressCallback(progHandle)

	var imagesOut *C.sd_image_t
	var numOut C.int
	ok := C.generate_image(c.handle, &params, &imagesOut, &numOut)
	// Register cleanup of the output buffer immediately so it is freed even
	// on the failure path.
	if imagesOut != nil {
		defer C.free_sd_images(imagesOut)
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
	if p.Prompt != "" {
		params.prompt = C.CString(p.Prompt)
		defer C.free(unsafe.Pointer(params.prompt))
	}
	if p.NegativePrompt != "" {
		params.negative_prompt = C.CString(p.NegativePrompt)
		defer C.free(unsafe.Pointer(params.negative_prompt))
	}
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
	params.vace_strength = C.float(p.VaceStrength)
	params.seed = C.int64_t(p.Seed)
	params.video_frames = C.int(p.VideoFrames)
	params.fps = C.int(p.FPS)
	params.vae_tiling_params = goTilingParamsToC(p.VAETilingParams)
	params.cache = goCacheParamsToC(p.Cache)

	progHandle := installProgressCallback(progress)
	defer clearProgressCallback(progHandle)

	var framesOut *C.sd_image_t
	var numFrames C.int
	var audioOut *C.sd_audio_t
	ok := C.generate_video(c.handle, &params, &framesOut, &numFrames, &audioOut)
	// Register cleanup immediately so buffers are freed even on the failure path.
	if framesOut != nil {
		defer C.free_sd_images(framesOut)
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

// ListDevices returns the number of available GPU devices.
func ListDevices() int {
	var count C.int
	C.sd_list_devices(&count)
	return int(count)
}

// Convert invokes SD.cpp's model format conversion (e.g. PyTorch -> safetensors/GGUF).
func Convert(inputPath, outputPath string, outputType SDType) error {
	cIn := C.CString(inputPath)
	defer C.free(unsafe.Pointer(cIn))
	cOut := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cOut))
	if ok := C.convert(cIn, cOut, C.sd_type_t(outputType), 0); !ok {
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
		width:   C.int(img.Width),
		height:  C.int(img.Height),
		channel: C.int(img.Channel),
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
	return C.sd_sample_params_t{
		sample_steps:                     C.int(p.SampleSteps),
		thread_count:                     C.int(p.ThreadCount),
		cfg_scale:                        C.float(p.CFGScale),
		guidance:                         C.float(p.Guidance),
		clip_skip:                        C.float(p.ClipSkip),
		sample_method:                    C.sd_sample_method_t(p.SampleMethod),
		schedule:                         C.sd_schedule_t(p.Schedule),
		flow_shift:                      C.float(p.FlowShift),
		old_rate_coef:                   C.float(p.OldRateCoef),
		omega_at_x0:                     C.float(p.OmegaAtX0),
		omega_at_xt:                     C.float(p.OmegaAtXt),
		omega_at_vt:                     C.float(p.OmegaAtVt),
		smpt_at:                          C.int(p.SmptAt),
		smpt_at_dynamic_thresholding_max: C.int(p.SmptAtDynamicThresholdingMax),
		vary_at_x0:                       C.float(p.VaryAtX0),
		vary_at_xt:                       C.int(p.VaryAtXt),
		x0_weight:                        C.float(p.X0Weight),
		xt_weight:                        C.int(p.XtWeight),
		eta:                              C.float(p.Eta),
		discrete_flow_shift:              C.int(p.DiscreteFlowShift),
		neg_tau_at_x0:                    C.int(p.NegTauAtX0),
		neg_tau_at_xt:                    C.float(p.NegTauAtXt),
		use_karras:                       C.int(boolToInt(p.UseKarras)),
		use_beta_dy_shift:                C.int(boolToInt(p.UseBetaDyShift)),
		beta_dy_shift_strength:           C.float(p.BetaDyShiftStrength),
	}
}

func goTilingParamsToC(p TilingParams) C.sd_tiling_params_t {
	return C.sd_tiling_params_t{
		enable:          C.int(boolToInt(p.Enable)),
		upscale_factor:  C.int(p.UpscaleFactor),
		strength:        C.float(p.Strength),
		denoise:         C.float(p.Denoise),
		scale_emphasis:  C.float(p.ScaleEmphasis),
	}
}

func goCacheParamsToC(p CacheParams) C.sd_cache_params_t {
	return C.sd_cache_params_t{
		no_unload:            C.bool(p.NoUnload),
		model_offload:        C.bool(p.ModelOffload),
		moe_boundary:         C.float(p.MoeBoundary),
		vace_strength:        C.float(p.VaceStrength),
		use_cache:            C.bool(p.UseCache),
		keep_model_loaded:    C.bool(p.KeepModelLoaded),
		neg_embd_mask:        C.bool(p.NegEmbdMask),
		use_guidance:         C.bool(p.UseGuidance),
		guidance_scale:       C.float(p.GuidanceScale),
		seed_frame_idx:       C.float(p.SeedFrameIdx),
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
