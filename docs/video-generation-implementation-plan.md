# SOTA Implementation Plan: Image & Video Generation via stable-diffusion.cpp

**Project:** aiollama (Ollama fork)
**Goal:** Add video generation and broaden image-model coverage on the
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) native
backend, **while retaining MLX** as the optimized macOS backend for the image and
safetensors-LLM models it already supports. Text generation stays on llama.cpp
(unchanged). Supported on **all** platforms: Linux, macOS, Windows.
**Author:** Engineering analysis
**Date:** 2026-07-16
**Status:** Draft for review

---

## 1. Executive Summary

This document is a State-of-the-Art (SOTA) implementation plan for adding
diffusion-based video generation and broad image-model coverage on a single new
native backend, stable-diffusion.cpp (SD.cpp), across all supported operating
systems, **while keeping MLX as the optimized native backend for the models it
already supports on macOS.**

### Architectural decision: keep MLX, add SD.cpp as a complementary backend

The fork currently carries **two** native inference stacks:

| Stack | Purpose | Backends | Platforms |
|-------|---------|----------|-----------|
| llama.cpp | Text (LLM) generation | CUDA, Metal, Vulkan, ROCm, CPU | Win, Linux, Mac |
| MLX | Image generation (Z-Image, FLUX.2) + experimental safetensors LLM text | Metal, CUDA | Mac (primary), CUDA (secondary) |

This plan **adds SD.cpp as a third, complementary native backend** rather than
removing MLX. SD.cpp covers everything MLX does not: video, the broad image-model
ecosystem (SDXL, SD3, Qwen-Image, Chroma, …), and cross-platform coverage on
Linux/Windows. MLX is retained where it has unique value. The resulting
architecture is a clean three-backend split:

| Stack | Purpose | Backends | Platforms |
|-------|---------|----------|-----------|
| llama.cpp | Text (LLM) generation | CUDA, Metal, Vulkan, ROCm, CPU | Win, Linux, Mac |
| MLX | Image gen for natively-supported models (Z-Image, FLUX.2) + safetensors LLM text | Metal, CUDA | Mac (primary), CUDA (secondary) |
| stable-diffusion.cpp | Video generation (all models) + image gen for models MLX does not support + image/video on Linux/Windows | CUDA, Metal, Vulkan, OpenCL, SYCL, CPU | Win, Linux, Mac |

### Why keep MLX instead of removing it

An earlier draft of this plan proposed removing MLX entirely and routing all
image generation through SD.cpp. Detailed analysis (see Section 11, "MLX retention
analysis") showed that a full removal is not justified in the near term because:

1. **MLX runs 9 safetensors LLM text architectures** (Qwen3.5, Gemma4,
   Cohere2-MoE, Laguna, GLM4-MoE-Lite, …) directly from checkpoints, with no
   GGUF conversion. SD.cpp is diffusion-only and cannot replace this. Removing
   MLX would drop a real, macOS-specific capability with no replacement.
2. **MLX has deep Metal optimizations** that SD.cpp's ggml-Metal backend does
   not: wired-memory pinning (Apple unified memory), graph compilation / closure
   fusion (JIT Metal kernels), `mlx_fast_*` fused kernels (RMSNorm, RoPE, SDPA),
   and zero-copy mmap of safetensors to GPU. No benchmark establishes that
   SD.cpp+Metal matches MLX+Metal for FLUX.2 / Z-Image on Apple Silicon.
3. **Video on macOS is CPU-bound regardless.** The WAN VAE supports only CUDA
   and CPU (not Metal), so SD.cpp video on macOS Metal falls back to CPU VAE
   either way. Removing MLX does not improve this and only removes the optimized
   image path.

MLX's sole downside is a larger maintenance surface (two diffusion stacks on
macOS). The hybrid approach accepts that cost in exchange for preserving unique
capabilities and proven Metal performance.

### Recommended dispatch strategy (hybrid)

| Request | Backend selected |
|---------|-------------------|
| Image gen, model natively supported by MLX (Z-Image, FLUX.2), on macOS | **MLX** (optimized Metal) |
| Image gen, model NOT supported by MLX (SDXL, SD3, Qwen-Image, …) | **SD.cpp** |
| Image gen on Linux/Windows | **SD.cpp** (MLX is macOS-relevant only) |
| Video gen (any platform, any model) | **SD.cpp** (only option) |
| Safetensors LLM text on macOS | **MLX** runner (preserved) |
| GGUF LLM text (any platform) | **llama.cpp** (unchanged) |

The existing scheduler dispatch in `server/sched.go` already supports this
coexistence: `IsDiffGen()` (SD.cpp) vs `IsMLX()` (safetensors) vs the default
llama.cpp (GGUF) path. No scheduler rearchitecture is required to keep MLX.

SD.cpp is a pure C/C++ implementation built on ggml — the same lineage as
llama.cpp — and shares its build system conventions (CMake, per-backend
GPU compilation, GGUF support). It complements MLX by covering what MLX does not:

- **Image models:** SD1.x/2.x, SDXL, SD3/3.5, FLUX.1/2, Qwen-Image, Z-Image,
  Chroma, LongCat, Krea2, HiDream, Ideogram4, and image-edit variants
  (FLUX-Kontext, Qwen-Image-Edit, LongCat-Edit, Boogu-Edit).
- **Video models:** WAN 2.1/2.2 (T2V, I2V, FLF2V, VACE, TI2V), LTX-2.3,
  LingBot-Video.
- **Backends:** CPU (AVX/AVX2/AVX512), CUDA, Vulkan, Metal, OpenCL, SYCL.
- **Weight formats:** PyTorch checkpoints (`.ckpt`/`.pth`/`.pt`), safetensors,
  GGUF — with a built-in convert mode.
- **Features:** LoRA, ControlNet (SD1.5), ADetailer, PhotoMaker, TAESD,
  ESRGAN upscale, VAE tiling, flash attention, LCM, negative prompts,
  cross-platform reproducibility (`--rng cuda/cpu`), PNG metadata embedding.

SD.cpp adds video, the broad image-model ecosystem, and Vulkan/OpenCL/SYCL
coverage that MLX never had. It coexists with MLX: where MLX has a native
implementation (Z-Image, FLUX.2 image on macOS), MLX is preferred for its
deep Metal optimizations; SD.cpp handles everything else (video, other image
models, and all image/video work on Linux/Windows).

### Key findings from the codebase analysis

The aiollama fork already ships a working image-generation subsystem under
`x/imagegen/`. Its structural patterns are reusable for the SD.cpp migration:

1. **Runner subprocess pattern.** `x/imagegen/server.go` implements
   `llm.LlamaServer` by spawning a child process
   (`ollama runner --imagegen-engine`) that exposes a local HTTP server. The
   parent forwards requests to the child. This is the architecture SD.cpp
   integration reuses.
2. **Scheduler integration.** `server/sched.go:594` dispatches to
   `imagegen.NewServer(modelName)` when `model.Config.Capabilities` contains
   `"image"`. A parallel `"video"` capability extends this.
3. **API surface.** `/v1/images/generations` and `/v1/images/edits` are wired
   (`server/routes.go:1916-1917`) through OpenAI-compatible middleware
   (`middleware/openai.go:601`).
4. **Manifest + blob storage.** `x/imagegen/manifest/` defines a manifest format
   and content-addressed blob store that SD.cpp models can reuse with an adapted
   component-file layout.
5. **Model capabilities system.** `types/model/config.go` defines `ConfigV2`
   with a `Capabilities []string` field. Adding `"video"` is config-layer only.

### Realistic effort estimate

| Scope | Effort | Notes |
|-------|-------|-------|
| SD.cpp build integration (CMake + FetchContent, all backends) | 2-3 weeks | Model on the llama.cpp backend build pattern |
| CGO binding package (`x/sdcpp`) | 1-2 weeks | New bridge, coexists with the MLX bridge |
| Unified runner (image + video, `x/diffgen/`) | 2-3 weeks | New runner alongside the retained MLX imagegen runner |
| API endpoints + middleware (image reuse + new video) | 1-2 weeks | Image path preserved; video is new |
| Scheduler + capabilities + memory estimation | 1 week | Add `"video"` capability path; keep MLX dispatch |
| Model import (safetensors/GGUF → manifest) | 1-2 weeks | Component-file manifest |
| CLI + progress UX for image + video | 1 week | New diffgen CLI; MLX CLI retained |
| Multi-backend testing (CUDA/Metal/Vulkan/CPU) | 2-3 weeks | Parallel with above |
| **Total (focused, CUDA+CPU first)** | **~2.5-3.5 months** | Video + broad image coverage; MLX retained |

---

## 2. Codebase Architecture (As-Is)

### 2.1 Directory map (relevant subsystems)

```
main.go                 # entry; cobra CLI
cmd/                    # CLI commands (cmd.go registers imagegen flags)
api/                    # public Go client + API types (GenerateRequest has Width/Height/Steps/Image)
server/                 # HTTP server, routes.go, sched.go (scheduler), create.go (model import)
                        #   + imageapi.go/imagefiles.go: dedicated OpenAI Images API handlers
openai/                 # OpenAI-compatible request/response types (ImageGenerationRequest, etc.)
middleware/             # gin middleware (chat/completions/embeddings; image middlewares REMOVED — see note §4.2)
types/model/config.go   # ConfigV2 (Capabilities, ModelFormat, Architecture)
x/
  imagegen/             # *** existing image-gen subsystem (MLX-based) — RETAINED for MLX-supported image models ***
    imagegen.go         # ImageModel interface + loadImageModel
    server.go           # Server (llm.LlamaServer) wraps MLX subprocess
    runner.go           # Execute() entry for `ollama runner --imagegen-engine`
    cli.go              # CLI: ollama run <img-model> "prompt"
    types.go            # Request/Response/ModelMode types
    image.go            # MLX Array → PNG / base64
    memory.go           # CheckPlatformSupport, DetectModelType
    manifest/           # per-tensor blob manifest + weights loader
    mlx/                # CGO bridge to MLX C library — RETAINED (MLX kept)
    models/
      flux2/            # FLUX.2 Klein model impl (MLX tensors) — RETAINED
      zimage/           # Z-Image model impl (MLX tensors) — RETAINED
    safetensors/        # safetensors parsing + LoadModule reflection loader
  mlxrunner/            # *** separate MLX-based LLM runner (text gen) — RETAINED (safetensors LLM on macOS) ***
  create/               # safetensors→manifest creation utilities
  safetensors/          # safetensors extraction
cmake/
  local.cmake           # superbuild: llama.cpp + MLX via ExternalProject/FetchContent (+ SD.cpp added)
  mlx/                  # MLX CMake subproject — RETAINED (MLX kept)
llama/                  # llama.cpp server subproject + compat (RETAINED for text)
discover/               # GPU detection (per-OS files)
CMakeLists.txt          # root orchestration
LLAMA_CPP_VERSION       # pinned llama.cpp ref (RETAINED)
MLX_VERSION, MLX_C_VERSION  # pinned MLX refs — RETAINED (MLX kept)
```

### 2.2 Existing image-gen request flow (MLX path, retained)

```
CLI (cmd.go:886 imagegen.RunCLI)
  → api.Client.Generate(/api/generate)
  → server.GenerateHandler (routes.go:254)
  → scheduler GetRunner (sched.go) — selects imagegen.NewServer if capability=="image" AND model is MLX-supported on macOS
  → x/imagegen.Server (server.go) spawns subprocess `ollama runner --imagegen-engine`
  → x/imagegen.Execute (runner.go) starts HTTP server in subprocess
  → Server.Completion (server.go:258) POSTs to child /completion
  → child handleImageCompletion (imagegen.go:64) streams ndjson {step,total} then {image}
```

This MLX path is **retained** for the image models MLX supports natively
(Z-Image, FLUX.2) on macOS. A new parallel SD.cpp path (Section 4, Phase 2)
handles video and the broader image-model set.

### 2.3 Text-gen flow (retained, unchanged)

```
ollama run <llm-model>
  → scheduler selects llama.cpp server (sched.go:581-583, newServerFn)
  → llama/server subprocess runs llama.cpp
  → token streaming via /completion
```

Text generation is entirely on llama.cpp and is **not** affected by this plan.
The MLX-based safetensors LLM runner (`x/mlxrunner/`) is a separate text path
that is **retained** — it runs 9 safetensors LLM architectures (Qwen3.5, Gemma4,
Cohere2-MoE, Laguna, GLM4-MoE-Lite, …) directly from checkpoints on macOS
without GGUF conversion. Keeping MLX preserves this capability; SD.cpp is
diffusion-only and cannot serve it.

### 2.4 Key interfaces and contracts

- `llm.LlamaServer` (`llm/`): the interface every runner implements. The imagegen
  `Server` implements it (`server.go:471`). The new SD.cpp runner must too.
- `ConfigV2.Capabilities` (`types/model/config.go:18`): `[]string` — currently
  uses `"image"`, `"completion"`, `"vision"`, `"audio"`, `"tools"`, etc.
- Scheduler dispatch (`sched.go:594`): `if slices.Contains(capabilities, "image")`.
- `ImageModel` interface (`imagegen.go:19`): MLX-specific (`*mlx.Array` return) —
  **retained** for the MLX image path. The new SD.cpp runner uses its own
  native interface (`DiffModel`); the two coexist.

### 2.5 Build system

`cmake/local.cmake` is a superbuild using `ExternalProject_Add`:
- llama.cpp (from `LLAMA_CPP_VERSION` pin) → `ollama_add_llama_server_build()`
- MLX + MLX-C (from `MLX_VERSION`/`MLX_C_VERSION` pins) → `ollama_add_mlx_build()` — **retained**
- SD.cpp (from `SD_CPP_VERSION` pin, added) → `ollama_add_sdcpp_build()` — **new**

Backends selected via `OLLAMA_LLAMA_BACKENDS` (cuda_v12, rocm_v7_1, vulkan, ...)
and `OLLAMA_MLX_BACKENDS` (cuda_v13, metal_v3/v4) — both **retained**. A new
`OLLAMA_SDCPP_BACKENDS` variable (cpu;cuda_v12;metal;vulkan) governs SD.cpp
backends. The three backend sets are independent and can be configured
separately per platform.

---

## 3. stable-diffusion.cpp Integration Target

### 3.1 SD.cpp C API (relevant surface)

From `include/stable-diffusion.h`:

```c
// Context creation
typedef struct {
    const char* model_path;              // diffusion model (safetensors/gguf)
    const char* clip_l_path;             // CLIP-L text encoder
    const char* clip_g_path;             // CLIP-G text encoder
    const char* clip_vision_path;        // for I2V (clip_vision_h)
    const char* t5xxl_path;              // UMT5-XXL encoder (WAN text)
    const char* vae_path;                // wan_2.1_vae / wan_2.2_vae / etc.
    const char* taesd_path;              // tiny VAE (low VRAM)
    const char* control_net_path;
    const char* backend;                 // "cuda" / "metal" / "vulkan" / "cpu"
    const char* params_backend;
    bool flash_attn, diffusion_flash_attn;
    bool vae_conv_direct, diffusion_conv_direct;
    enum sd_type_t wtype;
    int n_threads;
    bool enable_mmap;
    const char* max_vram;                // GiB budget for offload
    bool stream_layers;                  // residency+prefetch streaming
    // ... lora, embeddings, hires, photo_maker, pulid, etc.
} sd_ctx_params_t;

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params);
SD_API void free_sd_ctx(sd_ctx_t* ctx);
SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* ctx);
SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* ctx);

// Image generation
typedef struct {
    const char* prompt, *negative_prompt;
    int width, height;
    sd_sample_params_t sample_params;    // steps, cfg, sampler, scheduler
    int64_t seed;
    int batch_count;
    sd_image_t init_image;               // img2img
    sd_image_t mask_image;               // inpaint
    sd_image_t control_image;
    float control_strength;
    sd_tiling_params_t vae_tiling_params;
    // ... lora, pm_params, pulid, cache, hires
} sd_img_gen_params_t;

SD_API bool generate_image(sd_ctx_t* ctx, const sd_img_gen_params_t* params,
                           sd_image_t** images_out, int* num_images_out);

// Video generation
typedef struct {
    const char* prompt, *negative_prompt;
    sd_image_t init_image;               // I2V / TI2V reference frame
    sd_image_t end_image;                // FLF2V end frame
    sd_image_t* control_frames;          // VACE control
    int width, height;
    sd_sample_params_t sample_params;     // steps, cfg, sampler, scheduler, flow_shift
    sd_sample_params_t high_noise_sample_params;  // Wan2.2 dual-stage
    float moe_boundary, vace_strength;
    int64_t seed;
    int video_frames;                    // e.g. 33
    int fps;                             // output metadata
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
} sd_vid_gen_params_t;

SD_API bool generate_video(sd_ctx_t* ctx, const sd_vid_gen_params_t* params,
                           sd_image_t** frames_out, int* num_frames_out,
                           sd_audio_t** audio_out);

// Callbacks
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames,
                                bool is_noisy, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode,
                                    int interval, bool denoised, bool noisy, void* data);
SD_API void sd_cancel_generation(sd_ctx_t* ctx, enum sd_cancel_mode_t mode);
```

Key observations:
- `generate_image` and `generate_video` are **synchronous and blocking**. They
  allocate `images_out`/`frames_out` which the caller frees with `free_sd_images`.
- Progress and preview callbacks provide streaming hooks (step/frame updates).
- `sd_image_t` is `{width, height, channel, uint8_t* data}` — raw RGB.
- `max_vram` + `stream_layers` enable CPU offload for large models (critical
  for 14B video models on limited hardware).
- `sd_ctx_supports_image_generation` / `sd_ctx_supports_video_generation` allow
  runtime capability queries based on loaded components.

### 3.2 Model families supported (image + video)

**Image models:** SD1.x/2.x, SDXL, SD3/3.5, FLUX.1-dev/schnell, FLUX.2-dev/klein,
Lens, Chroma, Chroma1-Radiance, Qwen-Image, PiD, LongCat-Image, Z-Image, MiniT2I,
Ovis-Image, Anima, ERNIE-Image, Boogu-Image, Krea2, SeFi-Image, HiDream-O1-Image,
Ideogram4.

**Image-edit models:** FLUX.1-Kontext-dev, Qwen-Image-Edit, LongCat-Image-Edit,
Boogu-Image-Edit.

**Video models:** WAN 2.1/2.2 (T2V, I2V, FLF2V, VACE, TI2V), LTX-2.3, LingBot-Video.

### 3.3 Backends (all platforms)

| Backend | Platforms | Notes |
|---------|-----------|-------|
| CPU (AVX/AVX2/AVX512) | Win, Linux, Mac (x86) | Universal fallback |
| CUDA | Win, Linux | Primary GPU for video |
| Metal | Mac (arm64 + x86) | Replaces MLX on macOS |
| Vulkan | Win, Linux | Cross-vendor GPU |
| OpenCL | Win, Linux | Legacy GPU |
| SYCL | Linux | Intel GPUs |

### 3.4 Weight formats

- PyTorch checkpoint (`.ckpt`/`.pth`/`.pt`)
- Safetensors (`.safetensors`)
- GGUF (`.gguf`) — SD.cpp has a built-in `convert` mode to convert between
  formats, which can be used during `ollama create`.

---

## 4. Implementation Plan

### 4.1 Design principles

1. **SD.cpp is an added diffusion backend** for image **and** video, covering
   what MLX cannot: video, the broad image-model ecosystem (SDXL, SD3,
   Qwen-Image, …), and all platforms (Linux/Windows). **MLX is retained** as
   the optimized macOS backend for the image and safetensors-LLM models it
   already supports.
2. **Text generation stays on llama.cpp** — fully unchanged. The MLX
   safetensors LLM runner is also retained for the 9 text architectures it
   serves on macOS.
3. **New diffgen runner alongside the MLX runner.** A new `x/diffgen/`
   package handles image and video via SD.cpp, **coexisting** with the
   retained `x/imagegen/` (MLX) and `x/mlxrunner/` runners. The scheduler
   picks the backend per model. The new runner exposes `/completion`
   (streaming ndjson) like the existing imagegen runner, with mode detected
   from the loaded model.
4. **Capabilities: `"image"` and `"video"` are distinct.** A model is one or the
   other (or both, if the SD.cpp context supports both), determined at import
   time by `model_index.json` architecture. The scheduler dispatches
   accordingly — to MLX for MLX-supported image models on macOS, to SD.cpp
   for video and other image models.
5. **Mirror the proven runner pattern.** The new runner implements
   `llm.LlamaServer` and is spawned as a subprocess, exactly like the existing
   `x/imagegen/server.go`.
6. **Backend selection reuses existing discovery.** `discover/` detects
   CUDA/Metal/Vulkan/CPU devices. SD.cpp's `backend` string maps directly.
7. **Cross-platform by construction.** All target backends (CPU/CUDA/Metal/
   Vulkan) are supported on the relevant platforms from phase 0.

---

### Phase 0: Foundation and build integration

**Goal:** Build SD.cpp as a native library alongside llama.cpp **and MLX**,
without any Go integration yet. MLX build wiring is **retained**; SD.cpp is added
as a third native stack.

#### 0.1 Keep MLX in the build (no removal)
- **Retain** `cmake/mlx/` and the `ollama_add_mlx_build()` function in
  `cmake/local.cmake`.
- **Retain** `MLX_VERSION` and `MLX_C_VERSION` files.
- **Retain** `OLLAMA_MLX_BACKENDS` cache variable and `mlx_*` preset logic.
- **Retain** `x/imagegen/mlx/`, `x/mlxrunner/`, and MLX model implementations
  (`x/imagegen/models/flux2/`, `x/imagegen/models/zimage/`). These continue to
  serve MLX-supported image models (Z-Image, FLUX.2) and the 9 safetensors LLM
  text architectures on macOS.
- The new SD.cpp native implementations load model files directly (no Go-side
  model code needed), but they do **not** replace the MLX image path.

#### 0.2 Pin SD.cpp version
- Add `SD_CPP_VERSION` file at repo root (mirroring `LLAMA_CPP_VERSION`).
- Pin to a release/tag with WAN video support merged (PR #778 merged
  2025-09-06; the `vid_gen` async server API landed in commit 4d626d2,
  2026-04-18). Use the latest stable release or a recent `master` commit.

#### 0.3 CMake superbuild integration
Create `cmake/sdcpp/CMakeLists.txt` (mirroring the structure used by the
llama.cpp server subproject at `llama/server/CMakeLists.txt`):
- `FetchContent_Declare(stable-diffusion.cpp GIT_REPOSITORY ... GIT_TAG ${SD_CPP_VERSION})`
- Configure SD.cpp build options per backend:
  - CPU: default, with `GGML_CPU_ALL_VARIANTS` for AVX/AVX2/AVX512.
  - CUDA: `GGML_CUDA=ON`, forward `CMAKE_CUDA_ARCHITECTURES`.
  - Metal: `GGML_METAL=ON` (macOS), `GGML_METAL_EMBED_LIBRARY`.
  - Vulkan: `GGML_VULKAN=ON`.
  - OpenCL/SYCL: optional, behind explicit backend selection.
- Enable container output: `USE_WEBP`, `USE_WEBM` (for WebM/animated-WebP
  video containers when built with the library server).
- Install `libstable-diffusion.{so,dylib,dll}` + headers into
  `${OLLAMA_LIB_DIR}/sdcpp/<backend>/`.

Add to `cmake/local.cmake`:
- New `OLLAMA_SDCPP_BACKENDS` cache variable
  (e.g. `cpu;cuda_v12;metal;vulkan`).
- New `ollama_add_sdcpp_build()` function modeled on
  `ollama_add_llama_server_build()`.
- Wire into the `ollama-local` aggregate target.

#### 0.4 ggml coexistence strategy
SD.cpp vendors its own ggml; llama.cpp uses its own pinned ggml; MLX has its
own runtime. To avoid symbol clashes when multiple are loaded into the same
process (the Ollama binary links llama.cpp at build time and loads SD.cpp / MLX
as shared libs):
- **Build SD.cpp as a shared library** (`SD_BUILD_SHARED_LIBS=ON`) with hidden
  default visibility except the `SD_API` surface. SD.cpp already marks its API
  with `__attribute__((visibility("default")))` / `__declspec(dllexport)`.
- This keeps each ggml's internal symbols private to its shared object, avoiding
  conflicts. Verify with `nm`/`dumpbin` that no `ggml_*` symbols leak.
- **Phase 0 validation:** load `libllama`, `libstable-diffusion`, and the MLX
  library in a test binary and confirm no duplicate-symbol linker errors.

**Deliverable:** `cmake --build build` produces `libstable-diffusion.*` for the
selected backends alongside the llama.cpp runners **and the retained MLX
libraries**. MLX is preserved in the build.

---

### Phase 1: CGO binding package

**Goal:** A Go package `x/sdcpp/` that wraps the SD.cpp C API, as a **new** bridge
that coexists with the retained MLX bridge in `x/imagegen/mlx/`.

#### 1.1 Package structure
```
x/sdcpp/
  sdcpp.go          # CGO bridge: Context, GenerateImage, GenerateVideo, callbacks
  sdcpp_test.go     # binding round-trip tests
  types.go          # Go structs mirroring sd_*_params_t
  include/
    stable-diffusion.h  # vendored header (or symlink to SD.cpp source)
```

#### 1.2 CGO directives
Mirror the existing `x/imagegen/mlx/mlx.go` cgo pattern (which is retained):

```go
package sdcpp

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/include
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++
#include "stable-diffusion.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
```

#### 1.3 Go wrappers
Provide idiomatic Go wrappers for:
- `sd_ctx_params_init`, `new_sd_ctx`, `free_sd_ctx`
- `sd_img_gen_params_init`, `generate_image`, `free_sd_images`
- `sd_vid_gen_params_init`, `generate_video`
- `sd_set_progress_callback` / `sd_set_preview_callback` (with Go callback
  trampolines via `//export` or cgo trampolines — the MLX bridge used inline
  C callbacks; same approach)
- `sd_cancel_generation`
- `sd_ctx_supports_image_generation` / `sd_ctx_supports_video_generation`
- `sd_list_devices`
- `convert` / `convert_with_components` (for model import)

Map C types:
- `sd_ctx_t*` → opaque `Context` handle
- `sd_image_t` → `Image{Width, Height, Channels uint32; Data []byte}`
- `sd_img_gen_params_t` → Go struct `ImageGenParams`
- `sd_vid_gen_params_t` → Go struct `VideoGenParams`
- `sd_sample_params_t` → Go struct `SampleParams`
- `sd_audio_t` → `Audio` (stub for phase 1; LTXAV audio is future work)

**Deliverable:** `go build ./x/sdcpp/...` compiles against the installed
`libstable-diffusion`. A smoke test creates a context and frees it.

---

### Phase 2: Unified runner subprocess (image + video)

**Goal:** A working `ollama runner --diffgen-engine --model <name> --port <port>`
subprocess that can generate images **and** videos from SD.cpp models, streaming
progress. This runner is **new and parallel** to the retained MLX imagegen runner;
the scheduler selects which to spawn per model.

#### 2.1 New runner package: `x/diffgen/`
Sits alongside the retained `x/imagegen/` (MLX); does not replace it:

```
x/diffgen/
  diffgen.go        # DiffModel interface + loadModel (creates sdcpp.Context)
  server.go         # Server (llm.LlamaServer) wraps SD.cpp subprocess
  runner.go         # Execute() entry for `ollama runner --diffgen-engine`
  types.go          # DiffRequest / DiffResponse / ModelMode
  image.go          # sd_image_t → PNG / base64
  video.go          # frames → container (PNG stream / WebM / GIF / animated WebP)
  memory.go         # platform support, DetectModelType, backend selection
  manifest/         # component-file manifest loader
  cli.go            # CLI: ollama run <model> "prompt" (image or video) via SD.cpp
```

#### 2.2 Request/response types (`types.go`)
A unified request type handling both image and video (mode inferred from the
loaded model's capabilities, or explicit via a `Mode` field):

```go
type DiffRequest struct {
    Prompt          string   `json:"prompt"`
    NegativePrompt  string   `json:"negative_prompt,omitempty"`
    Mode            string   `json:"mode,omitempty"`  // "image" or "video"; auto-detected if empty

    // Common
    Width           int32    `json:"width,omitempty"`
    Height          int32    `json:"height,omitempty"`
    Steps           int      `json:"steps,omitempty"`
    Seed            int64    `json:"seed,omitempty"`
    CFGScale        float32  `json:"cfg_scale,omitempty"`
    Sampler         string   `json:"sampler,omitempty"`
    Scheduler       string   `json:"scheduler,omitempty"`
    OutputFormat    string   `json:"output_format,omitempty"` // image: "png"; video: "webm","webp","gif"
    Images          [][]byte `json:"images,omitempty"`  // init/control images (img2img, I2V)

    // Image-specific
    BatchCount      int      `json:"batch_count,omitempty"`
    ControlStrength float32  `json:"control_strength,omitempty"`

    // Video-specific
    VideoFrames     int      `json:"video_frames,omitempty"`  // e.g. 33
    FPS             int      `json:"fps,omitempty"`           // e.g. 16
    FlowShift       float32  `json:"flow_shift,omitempty"`    // WAN: 3.0
    EndImage        []byte   `json:"end_image,omitempty"`     // FLF2V end frame

    Options         *RequestOptions `json:"options,omitempty"`
}

type DiffResponse struct {
    Content  string `json:"content,omitempty"` // error text
    Image    string `json:"image,omitempty"`   // base64-encoded PNG (image mode)
    Video    string `json:"video,omitempty"`   // base64-encoded container (video mode)
    Done     bool   `json:"done"`
    Step     int    `json:"step,omitempty"`
    Total    int    `json:"total,omitempty"`
    Frame    int    `json:"frame,omitempty"`   // current frame (video mode)
    Frames   int    `json:"frames,omitempty"`  // total frames (video mode)
    StopReason string `json:"stop_reason,omitempty"`
}
```

#### 2.3 Runner `Execute()` (`runner.go`)
Mirror `x/imagegen/runner.go:Execute`. The subprocess:
1. Parses `--model`, `--port`, `--diffgen-engine` flags.
2. Loads the model via `loadModel()` (creates `sdcpp.Context` from manifest
   component paths + discovered backend).
3. Starts an HTTP server with `/health` and `/completion` handlers.
4. `/completion` handler calls `handleDiffCompletion()` which dispatches to
   `handleImageCompletion` or `handleVideoCompletion` based on model capabilities
   (queried via `sdcpp.SupportsVideoGeneration(ctx)`).

#### 2.4 `handleImageCompletion` (`diffgen.go`)
Mirrors `x/imagegen/imagegen.go:handleImageCompletion` (which is retained for
MLX), but calls SD.cpp:

```go
func (s *server) handleImageCompletion(w http.ResponseWriter, r *http.Request, req DiffRequest) {
    diffGenMu.Lock(); defer diffGenMu.Unlock()

    w.Header().Set("Content-Type", "application/x-ndjson")
    flusher := w.(http.Flusher)

    progress := func(step, steps int, time float32) {
        resp := DiffResponse{Step: step, Total: steps}
        json.NewEncoder(w).Encode(resp)
        w.Write([]byte("\n")); flusher.Flush()
    }
    sdcpp.SetProgressCallback(s.ctx, progress)

    images, err := sdcpp.GenerateImage(s.ctx, params)
    ...

    // Encode image as base64 PNG
    b64, err := EncodeImageBase64(images[0])
    ...

    resp := DiffResponse{Image: b64, Done: true}
    json.NewEncoder(w).Encode(resp)
    flusher.Flush()
}
```

#### 2.5 `handleVideoCompletion` (`diffgen.go`)
Video-specific handler:

```go
func (s *server) handleVideoCompletion(w http.ResponseWriter, r *http.Request, req DiffRequest) {
    diffGenMu.Lock(); defer diffGenMu.Unlock()

    w.Header().Set("Content-Type", "application/x-ndjson")
    flusher := w.(http.Flusher)

    progress := func(step, steps int, time float32) {
        resp := DiffResponse{Step: step, Total: steps}
        json.NewEncoder(w).Encode(resp)
        w.Write([]byte("\n")); flusher.Flush()
    }
    sdcpp.SetProgressCallback(s.ctx, progress)

    // Cancellation
    go func() {
        <-r.Context().Done()
        sdcpp.CancelGeneration(s.ctx)
    }()

    frames, err := sdcpp.GenerateVideo(s.ctx, params)
    ...

    // Encode frames → container (phase 1: PNG stream per frame)
    container, err := EncodeVideoBase64(frames, req.OutputFormat, req.FPS)
    ...

    resp := DiffResponse{Video: container, Done: true}
    json.NewEncoder(w).Encode(resp)
    flusher.Flush()
}
```

#### 2.6 Video encoding (`video.go`)
SD.cpp returns raw `sd_image_t` frames. Encoding options (in priority order):
- **PNG frame stream (phase 1 PoC):** stream `{frame: i, image: "base64png"}`
  per frame via ndjson, then `{done: true}`. No container dependency. Matches
  the existing imagegen streaming contract. Client reassembles.
- **WebM (VP8/VP9):** requires ffmpeg bindings (`github.com/3d0c/gmf`) or a
  pure-Go WebM muxer. Add in a later phase behind a build tag.
- **Animated WebP:** pure-Go via `golang.org/x/image/webp` (limited).
- **GIF:** pure-Go `image/gif` (acceptable for short previews).
- **AVI (MJPG):** if SD.cpp is built with WebM support, the library server can
  emit these; for the library path, encode in Go.

**Phase 1 decision:** PNG frame stream. Add container encoding (WebM) in
phase 5 once the pipeline is proven.

#### 2.7 Cancellation
Wire `sd_cancel_generation(ctx, SD_CANCEL_ALL)` to context cancellation
(shown in 2.5).

#### 2.8 Platform/backend selection (`memory.go`)
Mirror `x/imagegen/memory.go`. `CheckPlatformSupport()` now always returns nil
(SD.cpp supports all platforms). Backend selection:
```go
func ResolveBackend(gpus []ml.DeviceInfo) string {
    // Prefer CUDA > Metal > Vulkan > CPU based on discovered devices
    for _, g := range gpus {
        switch g.Library {
        case "cuda":  return "cuda"
        case "metal": return "metal"
        case "vulkan": return "vulkan"
        }
    }
    return "cpu"
}
```

**Deliverable:** `ollama runner --diffgen-engine --model <model> --port 9999`
runs; POSTing to `/completion` streams progress and returns images or frames.

---

### Phase 3: Scheduler and model dispatch

**Goal:** `ollama run <model> "prompt"` works end-to-end through the normal
server, with the scheduler managing the SD.cpp runner **alongside** the retained
MLX imagegen runner and the llama.cpp text runner.

#### 3.1 Capabilities: `"image"` and `"video"`
The `Capabilities` field already accepts arbitrary strings. Image models get
`["image"]`, video models get `["video"]`, and models supporting both (rare)
get `["image","video"]`. Set at import time (Phase 5). MLX-supported image
models (Z-Image, FLUX.2) keep `["image"]` and a `model_format: "mlx"` marker so
the scheduler can route them to MLX on macOS.

#### 3.2 Scheduler dispatch (`server/sched.go`)
At `sched.go:592-599`, **extend** the existing dispatch to add the SD.cpp
(diffgen) branch **without removing** the MLX imagegen and mlxrunner branches:

```go
switch {
case slices.Contains(req.model.Config.Capabilities, "video"):
    // SD.cpp is the only video backend (MLX has no video support)
    llama, err = diffgen.NewServer(modelName, "video")
case slices.Contains(req.model.Config.Capabilities, "image"):
    if isMLXSupportedImageModel(req.model) && runtime.GOOS == "darwin" {
        // Retained MLX path: Z-Image, FLUX.2 on macOS (optimized Metal)
        llama, err = imagegen.NewServer(modelName)
    } else {
        // SD.cpp path: all other image models, and image gen on Linux/Windows
        llama, err = diffgen.NewServer(modelName, "image")
    }
case isMLXSafetensorsLLM(req.model):
    // Retained MLX safetensors LLM runner (9 text architectures on macOS)
    llama, err = mlxrunner.NewClient(...)
default:
    // llama.cpp text path (existing newServerFn) for GGUF models
    config := llamaServerConfigForModel(req.model)
    llama, err = s.newServerFn(systemInfo, loadGpus, ...)
}
```

`isMLXSupportedImageModel` checks the model's architecture/format against the
MLX-supported set (Z-Image, FLUX.2) — this is a small, explicit allowlist. The
existing `imagegen.NewServer` and `mlxrunner.NewClient` branches are **kept**;
only the `"video"` capability and the SD.cpp `"image"` fallback are new.

#### 3.3 Runner ref additions
- Keep `runnerRef.isImagegen bool` (sched.go:1358) for the MLX path; add a
  parallel `isDiffgen bool` (or extend `runnerKind` to `"llama"`/`"mlx-image"`/
  `"mlx-llm"`/`"diffgen"`).
- Update `needsReload` (sched.go:1399) to check `wantImage || wantVideo`
  against the loaded runner kind, including the new diffgen kind.
- **Keep** the `mlxrunner` import (sched.go:27) and all `IsMLX()` checks in
  `server/routes.go` (lines 567, 1519, 2363, 2701) and `server/images.go:84` —
  they serve the retained MLX path. No removal.

#### 3.4 Memory estimation
SD.cpp context creation is where VRAM is consumed. `Server.Load()` must estimate
VRAM before spawning:
- Sum all component file sizes (diffusion_model + vae + text encoder + clip_vision)
  = weight footprint.
- Add an activation overhead factor:
  - Image: `~1.5×` weights (DiT activations during denoising).
  - Video: `~2-4×` weights (frame latents + temporal activations; 33 frames at
    832×480 latent ≈ small, but DiT peak is high).
- Use a conservative initial heuristic, refined by profiling. SD.cpp's own
  `max_vram` offload handles the gap.

#### 3.5 GPU/backend resolution
`discover/` returns `[]ml.DeviceInfo`. Map to SD.cpp `backend` string (see 2.8).
Add a `configureDiffgenSubprocessEnv` modeled on the **retained**
`configureMLXSubprocessEnv` (`server.go:185`) to set
`LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` to the sdcpp install dir for the selected
backend. The MLX env-configuration function stays for the MLX path.

**Deliverable:** `ollama run <model> "prompt"` → scheduler spawns the SD.cpp
runner → output streams back. Works for both image and video models. MLX image
and safetensors-LLM models continue to route to the retained MLX runners.

---

### Phase 4: API surface

**Goal:** HTTP API endpoints for image (preserved) and video (new).

#### 4.1 Native Ollama API
`api.GenerateRequest` already has `Width`/`Height`/`Steps`/`Image` fields.
Add video fields (additive, behind `omitempty` so existing clients unaffected):

```go
// Experimental: Video generation fields
NegativePrompt string  `json:"negative_prompt,omitempty"`
VideoFrames    int32   `json:"video_frames,omitempty"`
FPS            int32   `json:"fps,omitempty"`
CFGScale       float32 `json:"cfg_scale,omitempty"`
FlowShift      float32 `json:"flow_shift,omitempty"`
Sampler        string  `json:"sampler,omitempty"`
OutputFormat   string  `json:"output_format,omitempty"`
// InitImage/EndImage reuse existing Images []ImageData field
```

Add `Video string` to `api.GenerateResponse` for the final container. The
streaming `Step`/`Total` fields (already present) carry progress.

#### 4.2 Image endpoints (preserved, backend-agnostic)

> **SUPERSEDED (2026-07-19):** the image middlewares described below
> (`ImageGenerationsMiddleware`, `ImageEditsMiddleware`, `ImageWriter`) were
> **removed**. `/v1/images/generations` and `/v1/images/edits` are now
> dedicated handlers in `server/imageapi.go` (driving the scheduler
> directly) — required for spec conformance (n>1, multipart edits + mask,
> output transcoding, `usage`, `response_format=url`). The backend-agnostic
> routing property is preserved (the handlers use the same
> `scheduleRunner` path). See `docs/openai-api-conformance-report.md`.

`/v1/images/generations` and `/v1/images/edits` remain. The middleware
(`middleware/openai.go:601`) is unchanged — it converts to
`api.GenerateRequest` and the scheduler routes it to the appropriate runner
(MLX for MLX-supported models on macOS, SD.cpp otherwise). The middleware does
not need to know which backend is selected. `ImageWriter` works as-is.

#### 4.3 Video endpoints (new)
> **Status: SUPERSEDED.** This section described a non-spec-compliant
> `/v1/video/generations` surface that predates the official OpenAI Videos
> API. The OpenAI Videos API (Sora) **does** exist and is fully specified.
> The conformant implementation is documented in
> `docs/openai-videos-api-migration.md` and implemented as
> `POST/GET/DELETE /v1/videos`, `GET /v1/videos/{id}/content` (async job
> model with MP4 download). The text below is retained for historical
> reference only.
>
> OpenAI has no standardized video generation API as of 2026. Define an
> Ollama-native surface:

```
POST /v1/video/generations      # text-to-video
POST /v1/video/edits            # image-to-video (init_image in body)
```

Request schema (mirrors OpenAI image API conventions):
```json
{
  "model": "wan2.1-t2v-1.3b",
  "prompt": "a lovely cat playing",
  "negative_prompt": "low quality, blurry",
  "size": "832x480",
  "video_frames": 33,
  "fps": 16,
  "steps": 20,
  "cfg_scale": 6.0,
  "flow_shift": 3.0,
  "sampler": "euler",
  "output_format": "webm",
  "seed": 42,
  "stream": true
}
```

Response (non-streaming):
```json
{
  "created": 1704067200,
  "data": [{"video": "base64...", "format": "webm"}]
}
```

Streaming (SSE):
```
event: progress
data: {"step": 1, "total": 20}

event: frame
data: {"frame": 0, "image": "base64png..."}

event: done
data: {"created": 1704067200, "data": [{"video": "base64..."}]}
```

#### 4.4 Middleware (`middleware/`)
- `ImageGenerationsMiddleware` / `ImageEditsMiddleware` — **REMOVED
  (2026-07-19)**: superseded by the dedicated handlers in
  `server/imageapi.go`; see the SUPERSEDED note in §4.2.

Register in `routes.go`:
```go
r.POST("/v1/video/generations", cloudPassthroughMiddleware(...),
    middleware.VideoGenerationsMiddleware(), s.GenerateHandler)
r.POST("/v1/video/edits", cloudPassthroughMiddleware(...),
    middleware.VideoEditsMiddleware(), s.GenerateHandler)
```

**Deliverable:** `curl POST /v1/images/generations` and
`curl POST /v1/video/generations` both work with streaming progress.

---

### Phase 5: Model import and manifest

**Goal:** `ollama create <model>` from a directory of safetensors/gguf files
produces a manifest the runner can load, for both image and video models.

#### 5.1 Component-file manifest format
SD.cpp reads **whole checkpoint files** (not per-tensor splits like MLX). The
manifest stores each component as a single blob:

```json
{
  "schemaVersion": 2,
  "config": {"digest": "sha256:...", "size": 512},
  "layers": [
    {"mediaType": "application/vnd.ollama.image.model",
     "digest": "sha256:...", "size": 2600000000,
     "name": "diffusion_model"},
    {"mediaType": "application/vnd.ollama.image.model",
     "digest": "sha256:...", "size": 500000000,
     "name": "vae"},
    {"mediaType": "application/vnd.ollama.image.model",
     "digest": "sha256:...", "size": 9000000000,
     "name": "t5xxl"},
    {"mediaType": "application/vnd.ollama.image.model",
     "digest": "sha256:...", "size": 1700000000,
     "name": "clip_vision", "optional": true}
  ]
}
```

The config blob (`model_index.json`) declares architecture + capabilities:
```json
{
  "architecture": "WanVideoPipeline",
  "capabilities": ["video"],
  "model_format": "sdcpp",
  "components": ["diffusion_model", "vae", "t5xxl"],
  "optional_components": ["clip_vision", "high_noise_diffusion_model"],
  "defaults": {"width": 832, "height": 480, "video_frames": 33, "fps": 16, "flow_shift": 3.0}
}
```

For image models:
```json
{
  "architecture": "FluxPipeline",
  "capabilities": ["image"],
  "model_format": "sdcpp",
  "components": ["diffusion_model", "vae", "clip_l", "t5xxl"],
  "defaults": {"width": 1024, "height": 1024, "steps": 20}
}
```

#### 5.2 Import flow (`server/create.go` + `x/create/`)
Extend the model import. Detection: directory with `model_index.json` whose
`architecture` matches a known SD.cpp pipeline routes to the SD.cpp importer.
The import:
1. For each component file (`.safetensors`/`.gguf`/`.ckpt`), compute SHA256,
   copy to blob store, add manifest layer with `name = component`.
2. Optionally invoke SD.cpp `convert()` to normalize to safetensors/GGUF if the
   source is a PyTorch checkpoint.
3. Write `model_index.json` config blob with capabilities and architecture.
4. Write manifest.

Reuses the existing blob store (`envconfig.Models()/blobs`) and manifest dir.

#### 5.3 Manifest loader (`x/diffgen/manifest/`)
A simple loader (no per-tensor mmap needed — SD.cpp reads whole files):

```go
func (m *DiffManifest) ComponentPath(name string) (string, error) {
    for _, layer := range m.Manifest.Layers {
        if layer.Name == name {
            return m.BlobPath(layer.Digest), nil
        }
    }
    return "", fmt.Errorf("component %q not found", name)
}
```

#### 5.4 Model type detection (`x/diffgen/memory.go`)
Read `model_index.json`, check `architecture`:
- `"WanVideoPipeline"`, `"WanT2VPipeline"`, `"WanI2VPipeline"` → WAN video
- `"LTXVideoPipeline"` → LTX video
- `"FluxPipeline"`, `"StableDiffusionPipeline"`, `"SD3Pipeline"`,
  `"QwenImagePipeline"`, `"ZImagePipeline"`, etc. → image

**Deliverable:** `ollama create <model>` (from a dir with component files)
creates a runnable model for image or video.

---

### Phase 6: CLI and UX

**Goal:** `ollama run <model> "prompt"` produces image or video with a progress
bar, and `ollama run <model> -i image.png "prompt"` does img2img / I2V.

#### 6.1 CLI dispatch (`cmd/cmd.go`)
At `cmd.go:886`, **extend** the existing `imagegen.RunCLI` dispatch to also
handle SD.cpp diffgen models, without removing the MLX path:

```go
if diffgen.IsDiffModel(name) {
    return diffgen.RunCLI(cmd, name, opts.Prompt, interactive, opts.KeepAlive)
}
if imagegen.IsImageModel(name) {
    return imagegen.RunCLI(cmd, name, opts.Prompt, interactive, opts.KeepAlive) // retained MLX path
}
```

The diffgen runner auto-detects image vs video mode from the model's
capabilities. The `imagegen` check (MLX) remains for MLX-supported image models.

#### 6.2 `x/diffgen/cli.go`
Mirror `x/imagegen/cli.go` (which is **retained** for MLX models). Differences:
- Flags: `--width`, `--height`, `--steps`, `--seed`, `--negative`,
  `--cfg-scale`, `--sampler`, `--output-format`, plus video-specific
  `--video-frames`, `--fps`, `--flow-shift`.
- `-i`/`--init-image` for img2img (image) and I2V (video);
  `--end-image` for FLF2V (video).
- Progress: `progress.NewStepBar` shows step N/total; additionally show frame
  decode progress for video.
- Output: image → `<name>-<timestamp>.png`; video →
  `<name>-<timestamp>.webm` (or `.gif`/`.mp4` when container encoding is added).
  Attempt inline terminal preview only for images (terminal image protocols
  don't support video); for video, print the saved path.

#### 6.3 Interactive REPL
Mirror `x/imagegen/cli.go:runInteractive`. `/set` commands for both image and
video params:
```
/set width 1024
/set steps 20
/set frames 49        # video
/set fps 24            # video
/set cfg_scale 7.5
/set flow_shift 3.0    # video
```

**Deliverable:** Full CLI experience for image and video generation.

---

### Phase 7: Coexistence hardening and documentation

**Goal:** Ensure SD.cpp and MLX coexist cleanly in the same binary and build,
with no symbol clashes, correct scheduler routing, and up-to-date docs. MLX is
**retained** — this phase replaces the originally-planned "MLX removal" step.

#### 7.1 Verify MLX + SD.cpp coexistence
- Confirm `cmake --build build` produces both `libstable-diffusion.*` and the
  MLX libraries with no linker errors or duplicate `ggml_*` symbols (see Phase
  0.4 validation).
- Confirm the scheduler dispatch (Phase 3.2) routes MLX-supported image models
  to `imagegen.NewServer` and all other image/video models to
  `diffgen.NewServer`.
- Confirm the CLI dispatch (Phase 6.1) checks both `diffgen.IsDiffModel` and
  `imagegen.IsImageModel`.

#### 7.2 Keep MLX packages and build wiring (no removal)
- **Retain** `x/imagegen/` (MLX image-gen subsystem for Z-Image, FLUX.2 on macOS).
- **Retain** `x/imagegen/mlx/` (CGO bridge), `x/imagegen/models/flux2/`,
  `x/imagegen/models/zimage/` (MLX model implementations).
- **Retain** `x/mlxrunner/` (safetensors LLM runner — 9 text architectures).
- **Retain** `x/models/` MLX-dependent model implementations (qwen3_5,
  qwen3_5_moe, etc. that import `x/mlxrunner/`).
- **Retain** `x/internal/mlxthread/` (MLX thread affinity).
- **Retain** `cmake/mlx/`, `cmake/vendor-mlx-c-headers.cmake`,
  `ollama_add_mlx_build()`, `OLLAMA_MLX_BACKENDS`, `MLX_VERSION`,
  `MLX_C_VERSION`.

#### 7.3 Keep MLX references in Go code (no removal)
- `server/sched.go`: **keep** `mlxrunner` import (line 27), `IsMLX()` checks
  (lines 531, 1443), `mlxrunner.NewClient` branch (line 597). Add the diffgen
  branch alongside.
- `server/routes.go`: **keep** `IsMLX()` checks (lines 567, 1519, 2363, 2701).
- `server/images.go`: **keep** `IsMLX()` method (line 84) and its callers.
- `cmd/cmd.go`: **keep** `imagegen` import, `--imagegen` flag (line 2371),
  `use_imagegen_runner` option handling (line 222). Add the diffgen dispatch
  alongside.

#### 7.4 Update documentation
- `AGENTS.md`: update architecture section to reflect the **three-backend**
  model (llama.cpp for GGUF text, MLX for safetensors LLM text + MLX-supported
  image on macOS, SD.cpp for video + broad image coverage on all platforms).
- `x/diffgen/README.md`: document the SD.cpp runner and its coexistence with
  the MLX `x/imagegen/` runner.
- `docs/development.md`: add SD.cpp build instructions alongside the existing
  MLX instructions (both are built; `OLLAMA_SDCPP_BACKENDS` is new).

**Deliverable:** `go build ./...` and `cmake --build build` succeed with both
SD.cpp and MLX present. The scheduler correctly routes models to the right
backend. MLX image and safetensors-LLM capabilities are fully preserved.

---

### Phase 8: Multi-backend and hardening

**Goal:** Production-quality support across CPU, CUDA, Metal, Vulkan.

#### 8.1 Backend builds
In `cmake/local.cmake`, SD.cpp is built per backend variant:
- `cpu`: `GGML_CPU_ALL_VARIANTS=ON` (AVX/AVX2/AVX512) — universal.
- `cuda_v12`/`cuda_v13`: `GGML_CUDA=ON`, forward `CMAKE_CUDA_ARCHITECTURES`.
- `metal` (macOS arm64 + x86): `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY`.
- `vulkan`: `GGML_VULKAN=ON`.
- `opencl`/`sycl`: optional, behind explicit backend selection.

#### 8.2 VRAM offload
Wire SD.cpp's `max_vram` + `stream_layers` context params based on discovered
GPU free memory (`discover/`). For 14B models, default to
`max_vram = <freeVRAM - overhead>` with `stream_layers=true`.

#### 8.3 WAN VAE backend limitations
WAN VAE currently supports CUDA and CPU only (not Metal/Vulkan). Mitigations:
- On Metal/Vulkan, default to `--vae-on-cpu` (slow but functional).
- Gate WAN video models to recommend CUDA/CPU; show a warning on Metal/Vulkan
  with an estimated slowdown.
- Document the limitation; track SD.cpp upstream for Metal VAE support.

#### 8.4 Testing
- **Unit tests:** CGO binding round-trips, manifest parsing, request/response
  marshaling. Table-driven (see `format/bytes_test.go`).
- **Runner tests:** mock `sdcpp.Context` (or use a tiny CPU-only model) to test
  streaming and cancellation without GPU.
- **Integration tests** (`integration/`, behind `-tags=integration`):
  end-to-end generation with small models on CPU. Use **FLUX.2-Klein-4B**
  (Q2_K GGUF, 4-step turbo) for image and **WAN 2.2 T2V A14B** (Q2_K GGUF,
  dual-model MoE) for video. See Section 12.6 for exact weights, sources, and
  import commands. Gated behind `OLLAMA_TEST_DIFF_MODEL`.
- **Multi-backend CI matrix:** CUDA, Metal, Vulkan, CPU.

#### 8.5 Error handling
- SD.cpp returns `false` from `generate_image`/`generate_video` on failure.
  Capture stderr logs via the `sd_log_cb_t` callback and surface through the
  runner's `lastErr` mechanism (mirror `server.go:222 getLastErr`).
- Map common failures: OOM → `llm.ErrLoadRequiredFull` (reuse sched retry),
  unsupported backend → clear error message.

**Deliverable:** Builds for all backends; CI green; documented VRAM requirements
and backend limitations.

---

## 5. File-by-file change inventory

### New files
| Path | Purpose |
|------|---------|
| `SD_CPP_VERSION` | Pinned SD.cpp git ref |
| `cmake/sdcpp/CMakeLists.txt` | SD.cpp CMake subproject |
| `x/sdcpp/sdcpp.go` | CGO bridge to `libstable-diffusion` |
| `x/sdcpp/types.go` | Go structs mirroring `sd_*_params_t` |
| `x/sdcpp/sdcpp_test.go` | Binding tests |
| `x/sdcpp/include/stable-diffusion.h` | Vendored header |
| `x/diffgen/diffgen.go` | `DiffModel` interface, `loadModel` |
| `x/diffgen/server.go` | `Server` (llm.LlamaServer) — subprocess wrapper |
| `x/diffgen/runner.go` | `Execute()` — runner subprocess entry |
| `x/diffgen/types.go` | `DiffRequest` / `DiffResponse` |
| `x/diffgen/image.go` | `sd_image_t` → PNG / base64 |
| `x/diffgen/video.go` | Frame encoding (PNG stream / WebM / GIF) |
| `x/diffgen/memory.go` | Platform support, `DetectModelType`, backend selection |
| `x/diffgen/cli.go` | CLI for `ollama run <model>` (image + video) |
| `x/diffgen/manifest/manifest.go` | Component-file manifest loader |
| `x/diffgen/README.md` | Documentation |

### Deleted files
None. MLX is **retained** — no files are deleted. The new SD.cpp files are
additive alongside the existing MLX packages.

### Modified files
| Path | Change |
|------|-------|
| `CMakeLists.txt` | Include `cmake/sdcpp` if SD.cpp backends requested (additive; MLX include retained) |
| `cmake/local.cmake` | Add `ollama_add_sdcpp_build()`, `OLLAMA_SDCPP_BACKENDS` (additive; MLX build retained) |
| `server/sched.go` | Add diffgen dispatch for `"video"` and non-MLX `"image"`; keep mlxrunner/imagegen branches |
| `server/routes.go` | Register `/v1/video/generations`, `/v1/video/edits`; keep `IsMLX()` checks |
| `server/create.go` | Add SD.cpp model import path (detect + component blobs); keep MLX import path |
| `api/types.go` | Add video fields to `GenerateRequest`/`GenerateResponse` (additive) |
| `openai/openai.go` | Add `VideoGenerationRequest`/`VideoEditRequest` types + converters |
| `middleware/openai.go` | Add `VideoGenerationsMiddleware`/`VideoEditsMiddleware` + `VideoWriter` |
| `cmd/cmd.go` | Add `diffgen.RunCLI` dispatch; keep `imagegen.RunCLI` and `--imagegen` flag |
| `Dockerfile` | Add SD.cpp build deps (additive; MLX deps retained) |
| `AGENTS.md` | Update architecture section (three-backend model: llama.cpp + MLX + SD.cpp) |

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **ggml version skew** between SD.cpp's vendored ggml, llama.cpp's, and MLX's runtime | High | Build conflicts / symbol clashes | Build SD.cpp as a shared lib with hidden visibility (only `SD_API` exported). Verify no `ggml_*` symbols leak with `nm`/`dumpbin`. Phase 0 validation gate. MLX has its own runtime and does not share ggml symbols. |
| **WAN VAE CUDA/CPU only** (Metal/Vulkan unsupported) | High | macOS Metal / Vulkan users get slow video | Default to `--vae-on-cpu` on non-CUDA. Warn users. Track upstream for Metal VAE. Recommend CUDA for video. Note: this is unchanged from the removal plan — video on macOS Metal is CPU-bound regardless of whether MLX is present. |
| **14B model VRAM** exceeds typical consumer GPUs | High | OOM on load | Use SD.cpp `max_vram` + `stream_layers` offload. Default to 1.3B models in docs. Recommend Q8_0 GGUF quantization. |
| **Video container encoding** adds heavy Go deps (ffmpeg) | Medium | Binary bloat / cross-compilation pain | Phase 2 streams PNG frames (no container dep). Phase 6 adds WebM via optional cgo ffmpeg or pure-Go muxer behind a build tag. |
| **Synchronous generate calls block** the runner thread | Medium | Can't handle concurrent requests | Runner serializes per-model (existing `imageGenMu` pattern → `diffGenMu`). Scheduler serializes per runnerRef. Acceptable for phase 1. |
| **SD.cpp API instability** (README: "API may change frequently") | Medium | Binding breakage on version bump | Pin a specific commit. Wrap all C calls in a thin Go interface so binding changes are localized to `x/sdcpp/`. |
| **MLX + SD.cpp maintenance surface** (two diffusion stacks on macOS) | Medium | Higher maintenance cost | Accepted trade-off. MLX is retained for unique capabilities (safetensors LLM, deep Metal optimizations) that SD.cpp cannot replace. The scheduler already supports coexistence. Isolate SD.cpp in `x/diffgen/` and `x/sdcpp/` to minimize cross-contamination. |
| **ggml shared-lib symbol hiding** not sufficient on Windows | Low | DLL load failures | On Windows, SD.cpp uses `__declspec(dllexport)` for `SD_API` only; ensure the build sets `SD_BUILD_DLL` correctly. Test on Windows early in phase 0. |
| **WebM support not compiled** in SD.cpp build | Low | `output_format: "webm"` returns 400 | Detect at context creation. Fall back to PNG frame stream. |

---

## 7. Phased delivery schedule

| Phase | Deliverable | Dependencies | Duration |
|-------|------------|--------------|----------|
| **0** | SD.cpp builds via CMake (all backends) alongside retained MLX | None | 3 weeks |
| **1** | CGO bridge package `x/sdcpp/` compiles | Phase 0 | 2 weeks |
| **2** | Unified runner (`x/diffgen/`) generates image + video (manual model loading) | Phase 1 | 3 weeks |
| **3** | Scheduler dispatch + `ollama run` end-to-end (SD.cpp + MLX coexistence) | Phase 2 | 1 week |
| **4** | HTTP API endpoints (image preserved + video new) + streaming | Phase 3 | 2 weeks |
| **5** | Model import (`ollama create`) + manifest | Phase 3 | 2 weeks |
| **6** | CLI UX (progress bars, img2img/I2V, output formats) | Phase 4, 5 | 1 week |
| **7** | Coexistence hardening + documentation (MLX retained, no removal) | Phase 2, 6 | 1 week |
| **8** | Multi-backend (Metal/Vulkan), VRAM offload, tests | Phase 7 | 3 weeks |
| | | **Total** | **~18 weeks (~4-5 months)** |

**Phase 0-2 = functional PoC** (CUDA + CPU, 1.3B video + SD-turbo image, frame-stream output).
**Phase 3-6 = usable product** (API, CLI, model import, both image + video).
**Phase 7-8 = production quality** (MLX fully removed, all backends, tested).

---

## 8. Out of scope (future work)

- **Audio generation** (`sd_audio_t`, LTXAV). The C API exists but is separate
  from video frames. Future phase.
- **VACE / Fun control modes.** WAN VACE and FUN are in a separate SD.cpp PR.
  Add when SD.cpp ships them.
- **ControlNet / LoRA hot-swap** for image/video. SD.cpp supports
  `sd_ctx_load_control_net` and runtime LoRA; wire in a later phase if demand.
- **Streaming preview frames** via `sd_preview_cb_t`. Emits low-res intermediate
  frames during denoising (TAE/VAE preview). Useful for UX; defer.
- **Shared ggml linking.** Building SD.cpp and llama.cpp against one ggml to
  save disk/binary size. Only if symbol conflicts materialize despite shared-lib
  isolation.
- **OpenCL / SYCL backends.** SD.cpp supports them; add when demand exists.
- **Benchmarking MLX vs SD.cpp+Metal for image gen.** MLX is retained for its
  deep Metal optimizations, but no benchmark currently compares the two for
  FLUX.2/Z-Image on Apple Silicon. A future benchmark could determine whether
  SD.cpp+Metal ever matches or exceeds MLX+Metal, which would inform whether
  MLX image-gen could eventually be deprecated (safetensors LLM would still
  require MLX).

---

## 9. Key code references

| Concept | Location |
|---------|----------|
| Runner subprocess (llm.LlamaServer) pattern | `x/imagegen/server.go:35-154` (pattern reference; retained) |
| Runner entry point | `x/imagegen/runner.go:22-115` (pattern reference; retained) |
| Image completion streaming | `x/imagegen/imagegen.go:64-132` (pattern reference; retained) |
| Scheduler dispatch (image/mlx) | `server/sched.go:592-599` (to be extended with diffgen branch) |
| Scheduler MLX branch | `server/sched.go:531, 597` (retained) |
| Runner reload check | `server/sched.go:1393-1449` (to be modified to include diffgen kind) |
| OpenAI image middleware | `middleware/openai.go:601-680` (image: preserved) |
| OpenAI image types | `openai/openai.go:789-844` (image: preserved) |
| API types (Width/Height/Steps/Image) | `api/types.go:131-143`, `api/types.go:946` |
| Routes registration | `server/routes.go:1916-1917` |
| Model capabilities | `types/model/config.go:4-28` |
| Safetensors model import | `server/create.go:517-519` (to be extended for SD.cpp) |
| Manifest + blob storage | `x/imagegen/manifest/manifest.go` (pattern reference) |
| Model type detection | `x/imagegen/memory.go:53-80` (pattern reference; retained) |
| CLI dispatch | `cmd/cmd.go:886` (to be extended with diffgen dispatch) |
| CLI image-gen flow | `x/imagegen/cli.go:82-194` (pattern reference; retained) |
| Progress bar UX | `x/imagegen/cli.go:146-163` (pattern reference) |
| MLX CGO bridge (pattern, retained) | `x/imagegen/mlx/mlx.go:1-46` |
| MLX CMake (pattern, retained) | `cmake/local.cmake:142-195`, `x/imagegen/mlx/CMakeLists.txt` |
| llama.cpp backend build (pattern) | `cmake/local.cmake:363-450` (`ollama_add_llama_server_build`) |
| SD.cpp C API (image + video) | `include/stable-diffusion.h` (`generate_image`, `generate_video`, `sd_img_gen_params_t`, `sd_vid_gen_params_t`) |
| SD.cpp WAN docs | `docs/wan.md` (leejet/stable-diffusion.cpp) |

---

## 10. Summary

This plan adds video generation and broad image-model coverage via
stable-diffusion.cpp as a **new complementary backend**, while **retaining MLX**
as the optimized macOS backend for the image and safetensors-LLM models it
already supports. The resulting architecture is a three-backend split:

- **llama.cpp** → GGUF text generation (CUDA, Metal, Vulkan, ROCm, CPU).
- **MLX** → safetensors LLM text (9 architectures on macOS) + image gen for
  MLX-supported models (Z-Image, FLUX.2) on macOS (deep Metal optimizations).
- **stable-diffusion.cpp** → video generation (all models, all platforms) +
  image gen for models MLX does not support (SDXL, SD3, Qwen-Image, …) +
  image/video on Linux/Windows (CPU, CUDA, Metal, Vulkan, OpenCL, SYCL).

SD.cpp adds video (WAN 2.1/2.2, LTX-2.3, LingBot-Video) and the broad
image-model ecosystem that MLX never covered, plus Vulkan/OpenCL/SYCL coverage.
It does **not** replace MLX: MLX retains unique value in (1) running 9
safetensors LLM text architectures with no GGUF conversion, and (2) deep Metal
optimizations (wired memory, graph fusion, `mlx_fast_*` kernels) for image gen
on Apple Silicon that SD.cpp's ggml-Metal does not replicate. The scheduler
already supports this coexistence (`IsDiffGen()` vs `IsMLX()` vs llama.cpp).

The hard work is concentrated in three areas:
1. **Build integration + ggml coexistence** — getting SD.cpp to compile as an
   isolated shared library alongside llama.cpp **and MLX** without symbol
   clashes, across all backends.
2. **CGO binding** — wrapping the SD.cpp C API (`generate_image`,
   `generate_video`, callbacks, cancellation) in idiomatic Go.
3. **New diffgen runner** — a new subprocess runner handling both image and
   video modes via SD.cpp, alongside the retained MLX imagegen and mlxrunner.

Everything else — scheduler dispatch (extended, not replaced), manifest storage,
blob addressing, CLI framework, progress streaming, the existing OpenAI image
API — is adaptation of existing, tested code paths. A focused effort reaches a
functional PoC (CUDA + CPU, image + 1.3B video, frame-stream output) in ~8
weeks, and a production-quality multi-backend release with MLX retained in
~4-5 months.

---

## 11. MLX retention analysis

This section documents the analysis behind the decision to **keep MLX** rather
than remove it (as an earlier draft of this plan proposed).

### What MLX provides (and would be lost by removal)

| Capability | MLX | SD.cpp | Verdict |
|-----------|-----|--------|---------|
| Z-Image (image) | Full Go impl, optimized Metal | Supported natively | Replaceable, but Metal perf uncertain |
| FLUX.2 Klein (image + edit) | Full Go impl with img2img | Supported natively | Replaceable, but Metal perf uncertain |
| SDXL, SD3, Qwen-Image, Chroma, etc. | Not supported | Supported | SD.cpp superior |
| Video (WAN 2.1/2.2, LTX) | Not supported | Supported, but VAE Metal = slow (CPU fallback) | SD.cpp only option |
| 9 safetensors LLM text architectures | Experimental runner | Diffusion-only, cannot serve | **PERDU if MLX removed** |
| Deep Metal optimizations | Wired memory, graph fusion, `mlx_fast_*` | ggml Metal (different) | Performance uncertain |

### Three problems with full MLX removal

1. **Loss of safetensors LLM text (irreplaceable).** MLX runs 9 text model
   architectures directly from safetensors checkpoints without GGUF conversion.
   SD.cpp is diffusion-only and cannot replace this. The removal plan said
   "convert to GGUF" but that is a real loss of capability for macOS users of
   these experimental models.

2. **Uncertain Metal performance for image gen.** No benchmark compares
   SD.cpp+Metal vs MLX+Metal for FLUX.2/Z-Image. MLX has deep optimizations
   that SD.cpp lacks: wired-memory pinning (Apple unified memory), graph
   compilation / closure fusion (JIT Metal kernels), `mlx_fast_*` fused kernels
   (RMSNorm, RoPE, SDPA), and zero-copy mmap safetensors → GPU. SD.cpp uses
   ggml Metal which is functional but lacks this native integration. On Apple
   Silicon, MLX may be significantly faster for image generation.

3. **Video on macOS is slow regardless.** The WAN VAE supports only CUDA and
   CPU — not Metal. So even with SD.cpp, video on macOS Metal is degraded (VAE
   on CPU). Removing MLX does not change this, but it means macOS becomes a
   second-class platform for video either way.

### Recommended strategy: hybrid (keep MLX)

| Requested model | Capability | Backend |
|----------------|-----------|---------|
| Video (any) | video | SD.cpp (always) |
| Image, MLX-supported, macOS | image | MLX (optimized Metal) |
| Image, not MLX-supported | image | SD.cpp (broader model support) |
| Image, Linux/Windows | image | SD.cpp (MLX is macOS-relevant only) |
| Safetensors LLM text, macOS | completion | MLX runner (preserved) |
| GGUF LLM text (any) | completion | llama.cpp (unchanged) |

The existing scheduler dispatch (`server/sched.go`) already supports this
coexistence: `IsDiffGen()` (SD.cpp) vs `IsMLX()` (safetensors) vs llama.cpp
(GGUF). No scheduler rearchitecture is required to keep MLX.

### Tradeoff: maintenance vs performance

| Option | Maintenance | macOS performance | Text capability | Risk |
|--------|------------|-------------------|----------------|------|
| Full MLX removal (earlier draft) | Low (1 stack) | Uncertain | 9 archs lost | Metal perf regression |
| **Hybrid (keep MLX) — this plan** | 2 stacks | Optimized | Preserved | MLX maintenance cost |
| Hybrid simplified (keep MLX text only, image → SD.cpp) | Partial | Image uncertain | Preserved | Image perf compromise |

### Conclusion

The earlier Phase 7 (full MLX removal) is **not justified** in the near term.
It would lose 9 safetensors LLM text architectures without replacement, risk an
unmeasured image-performance regression on Apple Silicon, and its sole benefit
is simplified maintenance. The hybrid approach — keep MLX as the macOS native
backend for models it supports, use SD.cpp for video + unsupported image models
— preserves all capabilities at the cost of a larger maintenance surface. The
scheduler already supports coexistence.

If the priority is video, Phases 0-6 suffice — SD.cpp coexists with MLX out of
the box. The removal of MLX (Phase 7 as originally planned) adds nothing
functionally and risks regressions, so it is replaced by a coexistence-
hardening phase.

---

## 12. Implementation Status

**Date:** 2026-07-16
**Status:** Phases 0–8 implemented and committed. Go code compiles and unit
tests pass. Native build validation (12.1) is complete (CPU/Linux); E2E
testing and GPU-backend validation remain.

### Completed

| Phase | Commit | What was delivered |
|-------|--------|--------------------|
| 0 — Build integration | `ca50eac4` | `cmake/sdcpp/CMakeLists.txt`, `ollama_add_sdcpp_build()`, `OLLAMA_SDCPP_BACKENDS`, `SD_CPP_VERSION`, ggml shared-lib strategy |
| 1 — CGO binding | `ca50eac4` | `x/sdcpp/` (sdcpp.go, types.go, stable-diffusion.h, test helpers) |
| 2 — Runner | `ca50eac4` | `x/diffgen/` (runner.go, server.go, types.go, image.go, video.go, memory.go, manifest/) |
| 3 — Scheduler | `909d0fbd` | `server/sched.go` dispatch (`IsDiffGen()`), `runnerKindForModel()`, `server/images.go` `IsDiffGen()`, `needsReload` extended |
| 4 — API surface | `9fae26f4` | `/v1/video/generations`, `/v1/video/edits`, `middleware/openai.go` (`VideoGenerationsMiddleware`, `VideoEditsMiddleware`, `VideoWriter`), `openai/openai.go` types, `api/types.go` video fields |
| 5 — Model import | `909d0fbd` | `server/create.go` `convertFromSDCpp`, `detectModelTypeFromFiles` sdcpp detection, `x/diffgen/manifest/` component-file loader |
| 6 — CLI UX | `b14f815f` | `cmd/cmd.go` diffgen dispatch, `x/diffgen/cli.go` (flags, interactive REPL, progress bars), `x/diffgen/flags.go`, `x/diffutil/` shared helpers |
| 7 — Coexistence + docs | `9d594c10` | MLX fully retained (verified), `AGENTS.md` three-backend architecture, `x/diffgen/README.md`, `docs/development.md` SD.cpp build instructions |
| 8 — Multi-backend | `c4f931f6` | VRAM budget estimation + `--backend`/`--max-vram`/`--stream-layers` flags, `EstimateVRAMBudget` (multi-GPU, backend-scoped), `FormatVRAMGiB` (rounded), `ShouldStreamLayers` (size-gated), WAN VAE detection + per-request warning, OOM propagation via `DiffResponse.Error`, `sdcpp.SetLogCallback` → slog, `writeError` helper, unit tests + integration scaffold |

### Remaining work

The following items are explicitly listed in the plan but are **not yet
implemented**. They are non-blocking for a functional PoC but required for
production quality.

#### 12.1 Native build validation (Phase 0.4, 7.1) — DONE

- **`cmake --build build`** has been validated on Linux (WSL, GCC 13.3.0,
  CPU backend) for both SD.cpp (`OLLAMA_SDCPP_BACKENDS=cpu`) and llama.cpp
  (`ollama-local`). Both native stacks build and install cleanly in the same
  superbuild. CUDA/Metal/Vulkan toolchains were not available in this
  environment; the CPU path is validated end-to-end.
- **ggml symbol isolation** (Phase 0.4) is implemented and validated. The
  strategy embeds ggml+backends statically into `libstable-diffusion` (shared)
  with two isolation layers: (1) `-fvisibility=hidden` compile flags on the SD
  lib and its ggml deps, and (2) an ELF linker version script exporting only
  the `SD_API` surface (`sd_*`, `new_sd_ctx`, `free_sd_*`, `generate_image`,
  `generate_video`, `convert`). Verified with `nm -D`:
  `libstable-diffusion.so` exports **0** `ggml_*` symbols (was 495 before the
  fix), while `libggml-base.so` (llama.cpp) exports 535 — the two namespaces
  are now disjoint.
- **Phase 0.4 coexistence validation test binary** is written
  (`cmake/sdcpp/coexist_test.c`) and wired as the `sdcpp-coexist` CMake
  target (`cmake/local.cmake`). It links `libstable-diffusion` (SD.cpp) and
  `libggml-base` (llama.cpp) in the same process, resolves one symbol from
  each (`sd_get_system_info` and `ggml_type_name`), and confirms no
  duplicate-symbol linker errors. The test passes:
  ```
  SD.cpp: System Info: SSE3 = 1 | ... | AVX512_VNNI = 1
  llama.cpp ggml: ggml_type_name(0) = f32
  PASS: SD.cpp and llama.cpp ggml coexist without symbol conflicts
  ```
- **Go integration** validated: `go build -tags=sdcpp` links against the built
  `libstable-diffusion.so` and the `x/sdcpp` Go tests pass.

#### 12.2 Runner handler tests with mock (Phase 8.4) — DONE

- The HTTP handlers (`handleImageCompletion`, `handleVideoCompletion`) in
  `x/diffgen/runner.go` now have unit tests for streaming, cancellation, error
  propagation, OOM normalization, warning surfacing, mode override, and
  parameter forwarding (`x/diffgen/runner_test.go`, 25 tests).
- `sdcpp.Context` is abstracted behind the `sdContext` interface
  (`x/diffgen/context.go`) so a `mockSDContext` substitutes without a GPU or
  real model. `*sdcpp.Context` satisfies the interface via a compile-time
  assertion (`x/diffgen/context_sdcpp.go`). The test binary still links
  `libstable-diffusion` (cgo) but the mock never calls into C.
- Run with: `CGO_LDFLAGS='-L<libdir> -lstable-diffusion' go test -tags=sdcpp -run TestHandler ./x/diffgen/`

#### 12.3 Dockerfile (Section 5, modified files inventory) — DONE

- Added `sdcpp-cpu`, `sdcpp-cuda_v12`, `sdcpp-cuda_v13`, `sdcpp-vulkan` build
  stages (plus matching `publish-sdcpp-*` scratch stages) that build
  `libstable-diffusion` via `cmake/sdcpp` through the root superbuild
  (`-DOLLAMA_SDCPP_BACKENDS=<backend>`), mirroring the existing `mlx` stage
  pattern. These reuse the same toolchain stages already used for the
  llama-server GPU backends (`cpu-deps`, `cuda-12-deps`, `cuda-13-deps`,
  `vulkan-deps`) — CMake, Ninja, a C++ compiler, and the CUDA/Vulkan SDKs were
  already present in those stages, so no new system packages were required.
- Fixed a pre-existing gap in `cmake/local.cmake`'s superbuild guard: it
  required `OLLAMA_MLX_BACKENDS` to be set whenever `llama/server` was absent
  from the build context, with no equivalent allowance for
  `OLLAMA_SDCPP_BACKENDS`. This blocked an SD.cpp-only CMake invocation (the
  new Docker stages don't copy `llama/server`, matching how the `mlx` stage
  doesn't either). The guard now also accepts `OLLAMA_SDCPP_BACKENDS`.
- Added an explicit, additive `build-sdcpp` / `publish-go-sdcpp` target that
  builds the Go binary with `-tags=sdcpp` and links it against the `cpu`
  backend's `libstable-diffusion`, for maintainers/CI who want a
  diffgen-capable binary. This is **not** wired into the default
  `build`/`publish-go` target or any assembled image (`amd64`, `arm64`,
  `rocm`, `archive`, `image-archive`, or the final `ubuntu:24.04` stage),
  because `x/sdcpp` links `libstable-diffusion` directly (unlike the
  dlopen-based MLX bridge) — a default-enabled build would make every shipped
  `ollama` binary hard-depend on `libstable-diffusion.so` being resolvable at
  process startup on every platform/flavor (including e.g. the `rocm` image,
  which has no SD.cpp backend), which is a correctness risk beyond the scope
  of this item. Wiring SD.cpp into the default image/CI matrix is tracked
  separately (12.4).
- Not validated end-to-end in this environment (no Docker daemon / GPU
  toolchains available here, consistent with the 12.1 note that CUDA/Metal/
  Vulkan toolchains were unavailable for validation). The CMake invocations
  mirror the already-validated (12.1) local `cmake --build` flow for
  `OLLAMA_SDCPP_BACKENDS=cpu`.

#### 12.4 CI multi-backend matrix (Phase 8.4) — DEFERRED (low priority)

- No `.github/workflows/` configuration exists for a CUDA/Metal/Vulkan/CPU test
  matrix. The plan calls for this under Phase 8.4 ("Multi-backend CI matrix").
- The integration test scaffold (`integration/diffgen_test.go`) compiles and
  skips cleanly when `OLLAMA_TEST_DIFF_MODEL` is unset, but no CI job sets it.
- **Not a priority** for the current phase. The CPU backend is already
  validated locally (12.1) and the Dockerfile build stages are in place (12.3).
  A full multi-backend CI matrix (CUDA/Metal/Vulkan) is a regression-prevention
  measure, not a functional blocker — it will be implemented later once the
  E2E test harness (12.6) is wired up with real models, since a CI matrix
  without a runnable test adds little value over the existing local
  validation. Deferred until 12.6 lands and GPU-backed CI runners are available.

#### 12.5 Video container encoding (Phase 2.6) — DONE

- **WebM container encoding is implemented** (`x/diffgen/video.go`,
  `EncodeWebM`). Rather than vendoring a VP8/VP9 encoder or linking cgo
  ffmpeg bindings, raw RGB frames are piped through an external `ffmpeg`
  process (`rawvideo` in, `libvpx`/`webm` out), resolved once via
  `exec.LookPath("ffmpeg")` and cached. This keeps ffmpeg an **optional
  runtime dependency** (looked up on `PATH`), not a build-time Go module or
  cgo dependency — binaries built without ffmpeg present still build and run
  unchanged; they transparently fall back to the PNG frame-stream protocol.
  This directly addresses the "binary bloat / cross-compilation pain" risk
  in §6 without the correctness risk of a from-scratch VP8 encoder.
- `handleVideoCompletion` (`runner.go`) now checks
  `strings.EqualFold(req.OutputFormat, "webm") && SupportsContainerEncoding()`
  after generation completes. On success, it emits a single
  `DiffResponse{Video: base64(webm), Done: true, Frame: n, Frames: n}`
  message instead of streaming per-frame images (`Frame` is set equal to
  `Frames` so `Completed == Total` for progress-bar consumers, since no
  incremental frame progress is streamed on this path). On any failure
  (ffmpeg missing, encode error, or a cancelled request), it logs a warning,
  merges a "returned individual frames instead of a webm container" notice
  into the response `Warning` field (in addition to any existing
  WAN-VAE-fallback warning) so the degraded encoding is visible to callers
  and not just server logs, and falls back to the original per-frame PNG
  stream — fully backward compatible, and opt-in via `output_format`.
- ffmpeg's stdout is read into a **size-capped** buffer (512 MiB via
  `io.CopyN`, fully drained before `cmd.Wait()` per the `StdoutPipe`
  contract) rather than assigning an unbounded `bytes.Buffer` directly to
  `cmd.Stdout`, so a pathological frame count/size combination cannot grow
  the container-encoding response without limit; encoding fails cleanly with
  an error (and the usual frame-stream fallback) if the cap is exceeded.
- **`output_format` has no implicit default anywhere** — including on the
  OpenAI-compatible `/v1/video/generations` and `/v1/video/edits` endpoints.
  An earlier version of this change defaulted those two endpoints to
  `output_format="webm"` when unspecified, but that was reverted: since
  container encoding depends on an optional runtime dependency (ffmpeg on
  `PATH`), defaulting to it would have made the response shape
  (single video blob vs. last-frame PNG) depend on the server's incidental
  environment rather than an explicit, deployment-controlled choice. Callers
  that want a single WebM file must request `output_format: "webm"`
  explicitly; the native `/api/generate` path and CLI behave identically.
- **Bug fix in `middleware.VideoWriter`:** the writer previously collapsed
  `GenerateResponse.Video` and `.Image` into a single `lastImage` field and
  always re-emitted it as `.Image`, so `openai.ToVideoGenerationResponse`
  could never actually see a populated `.Video` field — a real webm
  container would have been silently downgraded to "last frame as PNG" over
  the OpenAI-compatible API. `VideoWriter` now tracks `lastVideo` and
  `lastImage` separately and restores the correct field before encoding the
  response, with `ToVideoGenerationResponse` setting `Format: "webm"` when a
  container is returned and `Format: "png"` for the frame fallback. This fix
  is independent of the output_format default question above and applies
  whenever a container is produced (i.e. `output_format="webm"` is
  requested explicitly).
- **Tests:** `x/diffgen/video_test.go` (`TestEncodeWebM*`, real ffmpeg
  subprocess, validates the EBML/Matroska magic header, dimension/channel
  validation, context cancellation, default FPS, the output size cap);
  `x/diffgen/runner_test.go` (`TestHandlerVideoWebMContainer`,
  `TestHandlerVideoWebMFallsBackWithoutOutputFormat`, full HTTP handler path
  with a mock `sdContext`, including the `Frame == Frames` progress
  invariant); `middleware/openai_test.go` (`TestVideoGenerationsMiddleware`,
  `TestVideoWriterResponse`, `TestVideoEditsMiddleware`, covering the
  no-implicit-default behavior and the container-vs-fallback-frame priority
  fix). All ffmpeg-dependent tests skip cleanly (`t.Skip`) when ffmpeg is not
  on `PATH`. Verified locally against the previously-built
  `build/lib/ollama/sdcpp/cpu/libstable-diffusion.so` (see §12.1) with
  ffmpeg 6.1.1 (libvpx-enabled) on WSL/Linux.
- **Not done:** animated WebP and GIF are still not produced server-side
  (GIF remains a client-side CLI fallback only); MP4/H.264 container output
  is not implemented (WebM/VP8 was prioritized as it's royalty-free and
  matches the plan's stated priority order). No CI job currently exercises
  the ffmpeg-dependent tests (they require ffmpeg installed on the runner);
  tracked alongside 12.4.

#### 12.6 End-to-end testing with real models — DONE (image + video E2E)

- The E2E test harness is implemented. The integration test scaffold
  (`TestDiffgenImageGeneration`, `TestDiffgenVideoGeneration`,
  `TestDiffgenVideoAPI`) now supports importing SD.cpp models from a **local
  directory** of component files via `OLLAMA_TEST_DIFF_MODEL_DIR`, in addition
  to the original registry-pull path. New tests were added:
  - `TestDiffgenImageGenerationProgress` — verifies streaming step/total
    progress events are emitted before the final image.
  - `TestDiffgenImportFromDirectory` — exercises the SD.cpp import path
    (`convertFromSDCpp`) end-to-end via `OLLAMA_TEST_DIFF_IMPORT_DIR` (uploads
    component files as blobs, creates the model, verifies it appears in
    `/api/tags` and `Show` returns a valid manifest). Does not require a GPU or
    real model — only tests the import/manifest path.
  - Helpers: `importDiffModelFromDir` (uploads blobs + `Create`, now
    recursive for subdirectory layouts like WAN 2.2's `LowNoise/`/`HighNoise/`),
    `ensureDiffModel` (import-or-pull dispatch), `sha256Digest`.
- Video E2E test params (width, height, steps, frames, fps, cfg_scale,
  flow_shift, timeout, size) are now configurable via `OLLAMA_TEST_DIFF_VIDEO_*`
  env vars. Defaults are CPU-friendly (832x480, 4 steps, 1 frame, 90 min
  timeout) so video E2E is tractable on CPU; GPU runners can raise the defaults
  (e.g. 832x480, 20 steps, 33 frames) for a fuller test.
- The `ollama create` path for SD.cpp models (`convertFromSDCpp`) now has full
  unit test coverage in `server/routes_create_test.go`:
  - `TestConvertFromSDCpp` (7 subtests): image model with components map,
    video model with dual diffusion models, filename-stem fallback naming,
    missing `model_index.json` error, invalid JSON error, base-layer
    append-order, model_index blob round-trip.
  - `TestCreateModelFromSDCppFilesSetsConfig`: full `CreateHandler` path —
    verifies `model_format: "sdcpp"`, `capabilities: ["video"]`, and that all
    diffusion component layers are present in the manifest with correct names
    and media types.
  - `TestCreateModelFromSDCppFilesImageDefaultsToImageCapability`: verifies
    the `["image"]` default when `model_index.json` omits capabilities.
  - `TestDetectModelTypeFromFiles` extended: sdcpp detection from
    `model_index.json` (path-agnostic via `filepath.Base`), and precedence over
    `.safetensors`.
- **Image E2E with real model — DONE.** `TestDiffgenImageGeneration` and
  `TestDiffgenImageGenerationProgress` pass against a real FLUX.2-Klein-4B
  model (Q4_0 diffusion + Q2_K text encoder + VAE) on CPU. The model loads
  via SD.cpp, text encoding completes (~32s), 4-step turbo sampling runs
  (~10 min on CPU), and the final image (~500-700 KB PNG) is returned with
  streaming progress events (26 events observed). Test timeouts raised to
  30 min for CPU. Run with:
  ```
  export OLLAMA_TEST_DIFF_MODEL=flux2-klein-4b
  export OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/flux2-klein-4b
  export OLLAMA_TEST_EXISTING=1  # use a pre-started server
  go test -tags=integration -run TestDiffgenImageGeneration ./integration/ -v -timeout 40m
  ```
- **Video E2E with real model — DONE.** `TestDiffgenVideoGeneration` and
  `TestDiffgenVideoAPI` pass against a real WAN 2.1 T2V 1.3B model (fp16
  diffusion + VAE + Q3_K_S t5xxl) on CPU. The lighter 1.3B single-stage model
  was used instead of the 14B dual-stage WAN 2.2 A14B (which would take hours
  per frame on CPU). The model loads via SD.cpp (~27 GB RAM for all
  components), text encoding completes (~80s), 4-step sampling runs (~374s),
  VAE decode completes (~55s), and the final frame (~245 KB PNG) is returned
  via the frame stream protocol. Total ~9 min/run on a 16-core CPU. Run with:
  ```
  export OLLAMA_TEST_DIFF_MODEL=wan2.1-t2v-1.3b
  export OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/wan2.1-t2v-1.3b
  export OLLAMA_TEST_EXISTING=1
  export OLLAMA_TEST_DIFF_VIDEO_FRAMES=1   # CPU-friendly; raise for GPU
  export OLLAMA_TEST_DIFF_VIDEO_STEPS=4
  go test -tags=integration -run TestDiffgenVideoGeneration ./integration/ -v -timeout 90m
  go test -tags=integration -run TestDiffgenVideoAPI ./integration/ -v -timeout 90m
  ```
  The `OLLAMA_TEST_DIFF_VIDEO_*` env vars allow tuning dimensions, steps,
  frame count, FPS, CFG scale, flow shift, and timeout for the target
  hardware. The test accepts both a single video container (when
  `output_format` is set) and individual PNG frames (frame stream protocol).
- **Import-path E2E with dummy fixture — DONE.** `TestDiffgenImportFromDirectory`
  passes against `integration/testdata/diffgen/flux2-dummy/` and
  `integration/testdata/diffgen/wan2.1-t2v-1.3b-dummy/` (dummy GGUF +
  safetensors files with valid magic bytes), validating the full import →
  manifest → Show path without a real model download. Run with:
  ```
  export OLLAMA_TEST_DIFF_IMPORT_DIR=integration/testdata/diffgen/wan2.1-t2v-1.3b-dummy
  go test -tags=integration -run TestDiffgenImportFromDirectory ./integration/ -v
  ```
- **GenerateHandler video dispatch fix.** The `GenerateHandler` previously
  only dispatched to `handleImageGenerate` for models with the `image`
  capability, so video models (capability `["video"]`) fell through to the
  completion path and returned `"%q does not support generate"`. The handler
  now also dispatches video-capable models to `handleImageGenerate`, and
  `handleImageGenerate` passes the correct capability (`video` vs `image`) to
  `scheduleRunner` so `CheckCapabilities` accepts it. The pre-loaded `*Model`
  is passed from `GenerateHandler` to `handleImageGenerate` to avoid a
  redundant `GetModel` call. `CheckCapabilities` was extended with
  `CapabilityVideo` → `errCapabilityVideo` in the capability error map.
- **Show handler video fix.** `GetModelInfo` previously returned early for
  `CapabilityImage` models (skipping GGUF tensor metadata loading), but video
  models fell through to `getModelData(m.ModelPath, ...)` with an empty
  `ModelPath` (sdcpp models have no `application/vnd.ollama.image.model`
  layer), causing `stat : no such file or directory` → 404. The early-return
  check now also covers `CapabilityVideo`, so `ollama show <video-model>`
  works correctly.

##### Test models (CPU-only, 2-bit quantized where available)

E2E tests run on CPU to avoid GPU dependencies in CI. Use the lowest-bit
quantized weights available (Q2_K) to keep download size and memory usage
minimal. Reference docs: [flux2.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md),
[wan.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/wan.md).

**Image — FLUX.2-Klein-4B** (turbo, 4-step generation):

| Component | File | Source |
|-----------|------|--------|
| diffusion_model | `flux-2-klein-4b-Q4_0.gguf` | [leejet/FLUX.2-klein-4B-GGUF](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/tree/main) |
| vae | `flux2-vae.safetensors` | [Comfy-Org/vae-text-encorder-for-flux-klein-4b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/tree/main/split_files/vae) |
| llm (text encoder) | `Qwen3-4B-Q2_K.gguf` | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main) |

Test params: `--width 512 --height 512 --steps 4 --cfg-scale 1.0`. The Klein-4B
model is a 4-step turbo model — fast enough for CPU CI. The GGUF repo only has
Q4_0 and Q8_0 (no Q2_K); Q4_0 is the smallest available (~2.5 GB). The text
encoder is Qwen3-4B (loaded via `--llm`, not `--clip_l`). The VAE from the
non-gated Comfy-Org repo is used to avoid the gated `black-forest-labs/FLUX.2-dev`
repo (which has `ae.safetensors`).

**Video — WAN 2.1 T2V 1.3B** (single-model, CPU-feasible — used for E2E):

| Component | File | Source |
|-----------|------|--------|
| diffusion_model | `wan2.1_t2v_1.3B_fp16.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models) |
| vae | `wan_2.1_vae.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae) |
| t5xxl | `umt5-xxl-encoder-Q3_K_S.gguf` | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf/tree/main) |

Test params (CPU): `--width 832 --height 480 --steps 4 --cfg-scale 6.0
--flow-shift 3.0 --video-frames 1 --fps 16` (1 frame, 4 steps — ~9 min on a
16-core CPU). The 1.3B model is a single-stage diffusion model (no
high-noise companion), so it loads faster and uses less RAM (~27 GB for all
components). Only fp16 safetensors are available for the 1.3B diffusion model
(~2.8 GB); the t5xxl text encoder uses Q3_K_S GGUF (~2.9 GB) to keep the text
encoder footprint smaller. The VAE is shared with WAN 2.1/2.2 14B models
(`wan_2.1_vae`, ~253 MB). Total download ~5.9 GB. This is the model used for
video E2E validation on CPU; for GPU-backed CI, raise frames/steps via
`OLLAMA_TEST_DIFF_VIDEO_*` env vars.

**Video — WAN 2.2 T2V A14B** (dual-model, MoE active ~2B — for GPU CI):

| Component | File | Source |
|-----------|------|--------|
| diffusion_model (low noise) | `Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf` | [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/tree/main/LowNoise) |
| high_noise_diffusion_model | `Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf` | [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/tree/main/HighNoise) |
| vae | `Wan2.1_VAE.safetensors` | [QuantStack/Wan2.2-T2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/tree/main/VAE) |
| t5xxl | `umt5-xxl-encoder-Q3_K_S.gguf` | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf/tree/main) |

Test params (GPU): `--width 832 --height 480 --steps 10 --cfg-scale 3.5
--flow-shift 3.0 --video-frames 33 --fps 16` (the SD.cpp example uses 33
frames). WAN 2.2 T2V A14B is a dual-stage model (LowNoise + HighNoise) — both
diffusion models must be present in the manifest. The VAE runs on CPU
(CUDA/CPU only, per Section 8.3). The t5xxl repo has no Q2_K; Q3_K_S is the
smallest available (~4.5 GB). The WAN repo stores files in `LowNoise/`,
`HighNoise/`, and `VAE/` subdirectories — flatten them into a single directory
for import (or use the recursive import helper). On CPU, this model is
impractically slow (~hours per frame); use the 1.3B model above for CPU E2E.

##### Importing the test models

> **See also:** `docs/creating-sdcpp-models.md` — a step-by-step user guide for
> creating SD.cpp-compatible models (WAN 2.2 T2V A14B and FLUX.2-Klein-4B),
> including the `model_index.json` contract, component-name reference, and
> troubleshooting.

```bash
# Image: FLUX.2-Klein-4B
ollama create flux2-klein-4b -f Modelfile.flux2-klein
# Modelfile points to a directory with:
#   model_index.json  (architecture: "FluxPipeline", capabilities: ["image"], model_format: "sdcpp")
#   flux-2-klein-4b-Q4_0.gguf          (component: "diffusion_model")
#   flux2-vae.safetensors               (component: "vae")
#   Qwen3-4B-Q2_K.gguf                  (component: "llm")

# Video: WAN 2.1 T2V 1.3B (CPU-feasible, used for E2E)
ollama create wan2.1-t2v-1.3b -f Modelfile.wan21
# Modelfile points to a directory with:
#   model_index.json  (architecture: "WanVideoPipeline", capabilities: ["video"], model_format: "sdcpp")
#   wan2.1_t2v_1.3B_fp16.safetensors    (component: "diffusion_model")
#   wan_2.1_vae.safetensors              (component: "vae")
#   umt5-xxl-encoder-Q3_K_S.gguf         (component: "t5xxl")

# Video: WAN 2.2 T2V A14B (dual-model, for GPU CI)
ollama create wan2.2-t2v-a14b -f Modelfile.wan22
# Modelfile points to a directory with:
#   model_index.json  (architecture: "WanVideoPipeline", capabilities: ["video"], model_format: "sdcpp")
#   Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf   (component: "diffusion_model")
#   Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf  (component: "high_noise_diffusion_model")
#   Wan2.1_VAE.safetensors                (component: "vae")
#   umt5-xxl-encoder-Q3_K_S.gguf          (component: "t5xxl")
```

Then run the integration tests:

```bash
# Image E2E (FLUX.2-Klein-4B, ~11 min CPU)
OLLAMA_TEST_DIFF_MODEL=flux2-klein-4b \
OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/flux2-klein-4b \
OLLAMA_TEST_EXISTING=1 \
go test -tags=integration -run TestDiffgenImageGeneration ./integration/ -v -timeout 40m

# Video E2E (WAN 2.1 T2V 1.3B, ~9 min CPU with 1 frame/4 steps)
OLLAMA_TEST_DIFF_MODEL=wan2.1-t2v-1.3b \
OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/wan2.1-t2v-1.3b \
OLLAMA_TEST_EXISTING=1 \
OLLAMA_TEST_DIFF_VIDEO_FRAMES=1 \
OLLAMA_TEST_DIFF_VIDEO_STEPS=4 \
go test -tags=integration -run TestDiffgenVideoGeneration ./integration/ -v -timeout 90m

# Video API E2E (same model, tests /v1/video/generations endpoint)
OLLAMA_TEST_DIFF_MODEL=wan2.1-t2v-1.3b \
OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/wan2.1-t2v-1.3b \
OLLAMA_TEST_EXISTING=1 \
OLLAMA_TEST_DIFF_VIDEO_FRAMES=1 \
OLLAMA_TEST_DIFF_VIDEO_STEPS=4 \
go test -tags=integration -run TestDiffgenVideoAPI ./integration/ -v -timeout 90m

# Import path E2E (dummy fixture, no download needed)
OLLAMA_TEST_DIFF_IMPORT_DIR=integration/testdata/diffgen/wan2.1-t2v-1.3b-dummy \
go test -tags=integration -run TestDiffgenImportFromDirectory ./integration/ -v
```

#### 12.7 Memory estimation refinement (Phase 3.4)

- The current VRAM estimation in `Server.Load()` uses `vramSize =
  TotalComponentSize()` (raw weight footprint) with no activation overhead
  factor. The plan calls for:
  - Image: ~1.5× weights (DiT activations).
  - Video: ~2–4× weights (frame latents + temporal activations).
- The pre-flight check compares `vramSize` against the budget, so models that
  fit in weights but OOM during activation are not caught early. SD.cpp's own
  `max_vram` offload handles the gap, but the heuristic should be refined by
  profiling.

### Summary of remaining work

| Item | Priority | Effort | Blocks | Status |
|------|----------|--------|--------|--------|
| Native build validation (`cmake --build`) | High | 1–2 days | E2E testing | **DONE** (CPU/Linux validated; CUDA/Metal/Vulkan pending toolchain) |
| ggml symbol isolation check | High | 0.5 day | Native build | **DONE** (0 leaked symbols; coexist test passes) |
| Runner handler tests (mock or CPU model) | Medium | 2–3 days | None | **DONE** (25 tests via mock sdContext interface) |
| Dockerfile SD.cpp deps | Medium | 0.5 day | Containerized CI | **DONE** (sdcpp-* build/publish stages added, reusing existing toolchain stages; opt-in `build-sdcpp`/`publish-go-sdcpp` Go target; not wired into default image) |
| CI multi-backend matrix | Low | 1–2 days | Regression prevention | **DEFERRED** (non-priority; revisit after 12.6 E2E harness and GPU CI runners) |
| Video container encoding (WebM) | Low | 3–5 days | Non-blocking (PNG stream works) | **DONE** (ffmpeg-based `EncodeWebM`, opt-in via `output_format: "webm"` everywhere including the HTTP video API — no implicit default; size-capped output buffer; `VideoWriter` container-priority bug fixed; fallback surfaced via response `Warning`) |
| E2E tests with real models (FLUX.2-Klein-4B + WAN 2.1 1.3B, CPU) | Medium | 1 day | Requires native build | **DONE** (image: `TestDiffgenImageGeneration` + `TestDiffgenImageGenerationProgress` pass with real FLUX.2-Klein-4B Q4_0 on CPU, ~11 min/run, 26 progress events; video: `TestDiffgenVideoGeneration` + `TestDiffgenVideoAPI` pass with real WAN 2.1 T2V 1.3B fp16 on CPU, ~9 min/run with 1 frame/4 steps, frame stream protocol; import: `TestDiffgenImportFromDirectory` passes with both flux2-dummy and wan2.1-t2v-1.3b-dummy fixtures; video E2E params configurable via `OLLAMA_TEST_DIFF_VIDEO_*` env vars) |
| `convertFromSDCpp` test fixture | Low | 0.5 day | Import path coverage | **DONE** (`TestConvertFromSDCpp` 7 subtests + `TestCreateModelFromSDCppFilesSetsConfig` + `TestCreateModelFromSDCppFilesImageDefaultsToImageCapability` + `TestDetectModelTypeFromFiles` sdcpp cases) |
| Memory estimation overhead factor | Low | 0.5 day | Pre-flight accuracy | Pending |

---

## 13. Reste à faire (TODO)

État consolidé au 2026-07-17, après revue du repo (commits jusqu'à
`b78b0f03`). Les Phases 0–8 du plan, la migration OpenAI Videos API Phase 1
(§4.3 superseded → `docs/openai-videos-api-migration.md`), les E2E image+vidéo
(§12.6) et l'encodage WebM (§12.5) sont **terminés**. Ce qui reste :

### 13.1 Migration API OpenAI Videos — Phase 2 (edits + extensions) — DONE

Source : `docs/openai-videos-api-migration.md` §5. Implémenté et testé (commit
à suivre). Conformité SDK OpenAI complète pour la création + édition +
extension de vidéos.

- [x] `DecodeVideoFrames(ctx, mp4 []byte, maxFrames int) ([]sdcpp.Image, error)`
      dans `x/diffgen/video.go` — décode un MP4/WebM stocké vers frames
      `sdcpp.Image` via ffmpeg (`image2pipe` PNG muxer). Tests round-trip
      (EncodeWebM → DecodeVideoFrames) + maxFrames cap.
- [x] `Transcoder.DecodeFrames` + `Transcoder.ConcatMP4` dans
      `server/videojobs/transcoder.go` (le worker n'importe pas x/diffgen,
      derrière le tag sdcpp). `DecodeFrames` retourne des PNG bytes ;
      `ConcatMP4` utilise le demuxer `concat` de ffmpeg avec fichiers
      temporaires (portable Windows — `ExtraFiles` non supporté).
- [x] `VideoEditHandler` (`POST /v1/videos/edits`) dans `server/videoapi.go` —
      multipart `prompt` + `video` (file part ou `{id}`). Résout la vidéo
      source en MP4 (job store par id, ou fichier uploadé), décode en frames,
      passe la **première** frame comme `req.Images[0]` (I2V) au worker.
      L'édition produit une **nouvelle** vidéo (re-render avec nouveau prompt),
      pas une concaténation (sémantique Sora). `Video.remixed_from_video_id`
      renseigné quand la source est un `{id}`.
- [x] `VideoExtendHandler` (`POST /v1/videos/extensions`) dans
      `server/videoapi.go` — multipart `prompt` + `seconds` (requis,
      `"4"`–`"20"`) + `video` (file ou `{id}`). Décode la source, prend la
      **dernière** frame comme init frame du segment d'extension, génère le
      segment via le worker. La réponse `Video.seconds` est le **total
      stitched** (source + extension) ; le contenu `/content` est le MP4
      concaténé via `ConcatMP4`.
- [x] Worker V2V/I2V-from-stored-video dans `server/videojobs/store.go` :
      `CreateParams.SourceVideoID` / `SourceVideo` / `Extend` /
      `SourceSeconds` ; dérivation init image (1ʳᵉ frame edit / dernière
      extend) ; `ConcatMP4` stitching ; `stitchSeconds` total.
- [x] Tests handler + intégration (mock + vrai modèle) — `server/videoapi_test.go`
      (8 tests edit/extend), `server/videojobs/store_test.go` (5 tests
      edit/extend worker + stitchSeconds), `server/videojobs/transcoder_test.go`
      (DecodeFrames round-trip + ConcatMP4), `x/diffgen/video_test.go`
      (DecodeVideoFrames round-trip + maxFrames), `integration/diffgen_test.go`
      (`TestDiffgenVideoEditAPI` E2E, gated `OLLAMA_TEST_DIFF_MODEL` + video
      capability).
- [x] Doc : §5 du rapport de migration mis à jour (checklist cochée + §5.5
      implementation notes) ; cette section §13.1.

### 13.2 Affinage estimation mémoire (Phase 3.4 / §12.7)

- [ ] `x/diffgen/server.go:72` utilise `vramSize = TotalComponentSize()` (poids
      bruts) sans facteur d'activation. Ajouter les facteurs prévus :
  - Image : ~1.5× poids (activations DiT pendant le denoising).
  - Vidéo : ~2–4× poids (latents de frames + activations temporelles).
- [ ] Affiner par profiling pour que la vérification pré-vol capture les modèles
      qui tiennent en poids mais OOM en activation. SD.cpp `max_vram` couvre le
      reste via offload.

### 13.3 Validation builds GPU (§12.1 — partiel)

- [ ] Le build CPU/Linux est validé. **Manquent** : CUDA, Metal, Vulkan
      (toolchains absentes sur l'environnement courant). Les fichiers CMake
      (`cmake/sdcpp/CMakeLists.txt`) et les stages Docker (`Dockerfile` sdcpp-*)
      sont en place mais non testés end-to-end sur ces backends.

### 13.4 CI multi-backends (§12.4 — DEFERRED)

- [ ] Pas de matrix CUDA/Metal/Vulkan/CPU dans `.github/workflows/`. Le scaffold
      d'intégration compile et skip proprement quand `OLLAMA_TEST_DIFF_MODEL` n'est
      pas set. Basse priorité : mesure anti-régression, pas un bloqueur
      fonctionnel. À revoir quand des runners GPU CI seront disponibles.

### 13.5 Variants de contenu `/content` (out-of-spec pour v1)

- [ ] `variant=thumbnail` et `variant=spritesheet` sur
      `GET /v1/videos/{id}/content` non implémentés (v1 retourne 501/404 pour
      ces variants ; seul `variant=video` — MP4 — fonctionne).

### 13.6 Hors-scope explicite (§8 du plan — non planifié)

Non implémentés et non prioritaires pour la v1 :

- Audio (`sd_audio_t` / Sora synced audio) — phase séparée.
- VACE / Fun control modes — en attente des PR SD.cpp correspondants.
- ControlNet / LoRA runtime hot-swap (image/vidéo) — `sd_ctx_load_control_net`
  et LoRA runtime existent côté SD.cpp, à câbler si demande.
- Preview frames streaming via `sd_preview_cb_t` (frames intermédiaires
  low-res pendant le denoising) — UX, différé.
- Shared ggml linking (SD.cpp + llama.cpp sur un ggml commun) — seulement si
  conflits de symboles malgré l'isolation shared-lib.
- Backends OpenCL / SYCL — à la demande.
- Benchmark MLX vs SD.cpp+Metal pour image gen — déterminerait si MLX image-gen
  pourrait un jour être déprécié (le LLM safetensors resterait sur MLX).

### 13.7 Résumé prioritaire

| # | Item | Priorité | Statut |
|---|------|----------|--------|
| 1 | OpenAI Videos Phase 2 (edits + extensions) | Moyenne | **DONE** (13.1) |
| 2 | Facteur d'activation estimation mémoire | Basse | À faire (§12.7) |
| 3 | Validation builds GPU (CUDA/Metal/Vulkan) | Basse | À faire (CPU validé) |
| 4 | CI multi-backends | Basse | DEFERRED |
| 5 | `variant=thumbnail`/`spritesheet` | Basse | Hors-scope v1 |
| 6 | Audio / VACE / ControlNet runtime / preview / OpenCL / SYCL | — | Hors-scope (§8) |

La conformité SDK OpenAI complète (création + édition + extension + polling +
download) est **en place**. Les items restants sont de la robustesse/CI/perf
(§13.2–13.4) ou explicitement hors-scope (§13.6).
