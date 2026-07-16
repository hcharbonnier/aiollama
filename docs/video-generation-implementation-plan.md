# SOTA Implementation Plan: Image & Video Generation via stable-diffusion.cpp

**Project:** aiollama (Ollama fork)
**Goal:** Unify all generative media (image **and** video) on the
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) native backend,
removing the existing MLX image-gen path entirely. Text generation stays on
llama.cpp (unchanged). Supported on **all** platforms: Linux, macOS, Windows.
**Author:** Engineering analysis
**Date:** 2026-07-16
**Status:** Draft for review

---

## 1. Executive Summary

This document is a State-of-the-Art (SOTA) implementation plan for consolidating
all diffusion-based generation (image + video) onto a single native backend,
stable-diffusion.cpp (SD.cpp), across all supported operating systems.

### Architectural decision: remove MLX, use SD.cpp for all media

The fork currently carries **two** native inference stacks:

| Stack | Purpose | Backends | Platforms |
|-------|---------|----------|-----------|
| llama.cpp | Text (LLM) generation | CUDA, Metal, Vulkan, ROCm, CPU | Win, Linux, Mac |
| MLX | Image generation (and experimental safetensors LLM) | Metal, CUDA | Mac (primary), CUDA (secondary) |

This plan **eliminates the MLX stack** and replaces it with SD.cpp for all
diffusion workloads (image **and** video). The result is a clean, two-backend
architecture:

| Stack | Purpose | Backends | Platforms |
|-------|---------|----------|-----------|
| llama.cpp | Text (LLM) generation | CUDA, Metal, Vulkan, ROCm, CPU | Win, Linux, Mac |
| stable-diffusion.cpp | Image **and** video generation | CUDA, Metal, Vulkan, OpenCL, SYCL, CPU | Win, Linux, Mac |

### Why SD.cpp is the right unified backend

SD.cpp is a pure C/C++ implementation built on ggml — the same lineage as
llama.cpp — and shares its build system conventions (CMake, per-backend
GPU compilation, GGUF support). It is a superset of what MLX currently provides
for image generation, plus native video support:

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

Because SD.cpp already supports Metal and CUDA, dropping MLX **loses nothing**
in image-gen capability while gaining video, Vulkan, OpenCL, and SYCL coverage,
and removing a large maintenance surface (the entire `x/imagegen/mlx/`,
`x/mlxrunner/`, and MLX CMake subproject).

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
| CGO binding package (`x/sdcpp`) | 1-2 weeks | Replace the MLX bridge |
| Unified runner (image + video, `x/diffgen/`) | 2-3 weeks | Replaces `x/imagegen/` |
| API endpoints + middleware (image reuse + new video) | 1-2 weeks | Image path mostly preserved; video is new |
| Scheduler + capabilities + memory estimation | 1 week | Add `"video"` capability path |
| Model import (safetensors/GGUF → manifest) | 1-2 weeks | Component-file manifest |
| CLI + progress UX for image + video | 1 week | Extend/replace `x/imagegen/cli.go` |
| MLX removal + migration cleanup | 1-2 weeks | Delete MLX packages, CMake, refs |
| Multi-backend testing (CUDA/Metal/Vulkan/CPU) | 2-3 weeks | Parallel with above |
| **Total (focused, CUDA+CPU first)** | **~3-4 months** | Full multi-backend release |

---

## 2. Codebase Architecture (As-Is)

### 2.1 Directory map (relevant subsystems)

```
main.go                 # entry; cobra CLI
cmd/                    # CLI commands (cmd.go registers imagegen flags)
api/                    # public Go client + API types (GenerateRequest has Width/Height/Steps/Image)
server/                 # HTTP server, routes.go, sched.go (scheduler), create.go (model import)
openai/                 # OpenAI-compatible request/response types (ImageGenerationRequest, etc.)
middleware/             # gin middleware: ImageGenerationsMiddleware, ImageEditsMiddleware
middleware/openai.go    # ImageWriter translates ndjson stream → OpenAI response
types/model/config.go   # ConfigV2 (Capabilities, ModelFormat, Architecture)
x/
  imagegen/             # *** existing image-gen subsystem (MLX-based) — TO BE REPLACED ***
    imagegen.go         # ImageModel interface + loadImageModel
    server.go           # Server (llm.LlamaServer) wraps MLX subprocess
    runner.go           # Execute() entry for `ollama runner --imagegen-engine`
    cli.go              # CLI: ollama run <img-model> "prompt"
    types.go            # Request/Response/ModelMode types
    image.go            # MLX Array → PNG / base64
    memory.go           # CheckPlatformSupport, DetectModelType
    manifest/           # per-tensor blob manifest + weights loader
    mlx/                # *** CGO bridge to MLX C library — TO BE DELETED ***
    models/
      flux2/            # FLUX.2 Klein model impl (MLX tensors) — TO BE DELETED
      zimage/           # Z-Image model impl (MLX tensors) — TO BE DELETED
    safetensors/        # safetensors parsing + LoadModule reflection loader
  mlxrunner/            # *** separate MLX-based LLM runner (text gen) — TO BE DELETED ***
  create/               # safetensors→manifest creation utilities
  safetensors/          # safetensors extraction
cmake/
  local.cmake           # superbuild: llama.cpp + MLX via ExternalProject/FetchContent
  mlx/                  # MLX CMake subproject — TO BE DELETED
llama/                  # llama.cpp server subproject + compat (RETAINED for text)
discover/               # GPU detection (per-OS files)
CMakeLists.txt          # root orchestration
LLAMA_CPP_VERSION       # pinned llama.cpp ref (RETAINED)
MLX_VERSION, MLX_C_VERSION  # pinned MLX refs — TO BE DELETED
```

### 2.2 Existing image-gen request flow (to be replaced)

```
CLI (cmd.go:886 imagegen.RunCLI)
  → api.Client.Generate(/api/generate)
  → server.GenerateHandler (routes.go:254)
  → scheduler GetRunner (sched.go) — selects imagegen.NewServer if capability=="image"
  → x/imagegen.Server (server.go) spawns subprocess `ollama runner --imagegen-engine`
  → x/imagegen.Execute (runner.go) starts HTTP server in subprocess
  → Server.Completion (server.go:258) POSTs to child /completion
  → child handleImageCompletion (imagegen.go:64) streams ndjson {step,total} then {image}
```

### 2.3 Text-gen flow (retained, unchanged)

```
ollama run <llm-model>
  → scheduler selects llama.cpp server (sched.go:581-583, newServerFn)
  → llama/server subprocess runs llama.cpp
  → token streaming via /completion
```

Text generation is entirely on llama.cpp and is **not** affected by removing MLX.
The MLX-based safetensors LLM runner (`x/mlxrunner/`) is a separate experimental
text path that is also removed; safetensors LLM models are not part of the
image/video scope and would fall back to conversion to GGUF or remain unsupported
(out of scope for this plan).

### 2.4 Key interfaces and contracts

- `llm.LlamaServer` (`llm/`): the interface every runner implements. The imagegen
  `Server` implements it (`server.go:471`). The new SD.cpp runner must too.
- `ConfigV2.Capabilities` (`types/model/config.go:18`): `[]string` — currently
  uses `"image"`, `"completion"`, `"vision"`, `"audio"`, `"tools"`, etc.
- Scheduler dispatch (`sched.go:594`): `if slices.Contains(capabilities, "image")`.
- `ImageModel` interface (`imagegen.go:19`): MLX-specific (`*mlx.Array` return) —
  replaced by an SD.cpp-native interface.

### 2.5 Build system

`cmake/local.cmake` is a superbuild using `ExternalProject_Add`:
- llama.cpp (from `LLAMA_CPP_VERSION` pin) → `ollama_add_llama_server_build()`
- MLX + MLX-C (from `MLX_VERSION`/`MLX_C_VERSION` pins) → `ollama_add_mlx_build()` — **removed**

Backends selected via `OLLAMA_LLAMA_BACKENDS` (cuda_v12, rocm_v7_1, vulkan, ...)
and `OLLAMA_MLX_BACKENDS` (cuda_v13, metal_v3/v4). The MLX variable is removed;
a new `OLLAMA_SDCPP_BACKENDS` variable governs SD.cpp backends.

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

1. **SD.cpp is the single diffusion backend** for image **and** video on all
   platforms. No MLX dependency remains.
2. **Text generation stays on llama.cpp** — fully unchanged.
3. **Unified diffgen runner.** A single new `x/diffgen/` package handles both
   image and video via SD.cpp, replacing `x/imagegen/` and `x/mlxrunner/`.
   The runner exposes `/completion` (streaming ndjson) like the existing imagegen
   runner, with mode detected from the loaded model.
4. **Capabilities: `"image"` and `"video"` are distinct.** A model is one or the
   other (or both, if the SD.cpp context supports both), determined at import
   time by `model_index.json` architecture. The scheduler dispatches accordingly.
5. **Mirror the proven runner pattern.** The new runner implements
   `llm.LlamaServer` and is spawned as a subprocess, exactly like the existing
   `x/imagegen/server.go`.
6. **Backend selection reuses existing discovery.** `discover/` detects
   CUDA/Metal/Vulkan/CPU devices. SD.cpp's `backend` string maps directly.
7. **Cross-platform by construction.** All target backends (CPU/CUDA/Metal/
   Vulkan) are supported on the relevant platforms from phase 0.

---

### Phase 0: Foundation and build integration

**Goal:** Build SD.cpp as a native library alongside llama.cpp, without any Go
integration yet. Remove MLX build wiring.

#### 0.1 Remove MLX from the build
- Delete `cmake/mlx/` and the `ollama_add_mlx_build()` function in
  `cmake/local.cmake`.
- Remove `MLX_VERSION` and `MLX_C_VERSION` files.
- Remove `OLLAMA_MLX_BACKENDS` cache variable and all `mlx_*` preset logic.
- Remove the `ollama-mlx-generate-wrappers` custom target and
  `cmake/vendor-mlx-c-headers.cmake`.
- Remove `x/imagegen/mlx/`, `x/mlxrunner/`, and MLX model implementations
  (`x/imagegen/models/flux2/`, `x/imagegen/models/zimage/`) — these are
  replaced by SD.cpp native implementations (no Go-side model code needed;
  SD.cpp loads the model files directly).

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
SD.cpp vendors its own ggml; llama.cpp uses its own pinned ggml. To avoid symbol
clashes when both are loaded into the same process (the Ollama binary links
llama.cpp at build time and loads SD.cpp as a shared lib):
- **Build SD.cpp as a shared library** (`SD_BUILD_SHARED_LIBS=ON`) with hidden
  default visibility except the `SD_API` surface. SD.cpp already marks its API
  with `__attribute__((visibility("default")))` / `__declspec(dllexport)`.
- This keeps each ggml's internal symbols private to its shared object, avoiding
  conflicts. Verify with `nm`/`dumpbin` that no `ggml_*` symbols leak.
- **Phase 0 validation:** load both `libllama` and `libstable-diffusion` in a
  test binary and confirm no duplicate-symbol linker errors.

**Deliverable:** `cmake --build build` produces `libstable-diffusion.*` for the
selected backends alongside the llama.cpp runners. MLX is gone from the build.

---

### Phase 1: CGO binding package

**Goal:** A Go package `x/sdcpp/` that wraps the SD.cpp C API, replacing the
MLX bridge.

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
Mirror the (now-deleted) `x/imagegen/mlx/mlx.go` cgo pattern:

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
progress.

#### 2.1 New runner package: `x/diffgen/`
Replaces `x/imagegen/` entirely:

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
  cli.go            # CLI: ollama run <model> "prompt" (image or video)
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
Replaces `x/imagegen/imagegen.go:handleImageCompletion`, but calls SD.cpp:

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
server, with the scheduler managing the SD.cpp runner.

#### 3.1 Capabilities: `"image"` and `"video"`
The `Capabilities` field already accepts arbitrary strings. Image models get
`["image"]`, video models get `["video"]`, and models supporting both (rare)
get `["image","video"]`. Set at import time (Phase 5).

#### 3.2 Scheduler dispatch (`server/sched.go`)
At `sched.go:592-599`, replace the MLX/imagegen branch with SD.cpp dispatch:

```go
switch {
case slices.Contains(req.model.Config.Capabilities, "video"):
    llama, err = diffgen.NewServer(modelName, "video")
case slices.Contains(req.model.Config.Capabilities, "image"):
    llama, err = diffgen.NewServer(modelName, "image")
default:
    // llama.cpp text path (existing newServerFn)
    config := llamaServerConfigForModel(req.model)
    llama, err = s.newServerFn(systemInfo, loadGpus, ...)
}
```

This removes the `imagegen.NewServer` and `mlxrunner.NewClient` branches
(sched.go:595, 597).

#### 3.3 Runner ref cleanup
- Replace `runnerRef.isImagegen bool` (sched.go:1358) with `runnerKind string`
  (`"llama"`/`"diffgen"`), or add `isDiffgen bool`.
- Update `needsReload` (sched.go:1399) to check `wantImage || wantVideo` against
  the loaded runner kind.
- Remove `mlxrunner` import (sched.go:27) and all `IsMLX()` checks in
  `server/routes.go` (lines 567, 1519, 2363, 2701) and `server/images.go:84`.

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
Reuse the `configureMLXSubprocessEnv` pattern (`server.go:185`) as
`configureDiffgenSubprocessEnv` to set `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`
to the sdcpp install dir for the selected backend.

**Deliverable:** `ollama run <model> "prompt"` → scheduler spawns the SD.cpp
runner → output streams back. Works for both image and video models.

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

#### 4.2 Image endpoints (preserved, rewired)
`/v1/images/generations` and `/v1/images/edits` remain. The middleware
(`middleware/openai.go:601`) is unchanged — it converts to
`api.GenerateRequest` and the runner handles it. The only change is the runner
behind it (SD.cpp instead of MLX). `ImageWriter` works as-is.

#### 4.3 Video endpoints (new)
OpenAI has no standardized video generation API as of 2026. Define an
Ollama-native surface:

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
- `ImageGenerationsMiddleware` / `ImageEditsMiddleware` — unchanged (already
  produce `api.GenerateRequest` with `Width`/`Height`).
- Add `VideoGenerationsMiddleware()` and `VideoEditsMiddleware()` mirroring
  the image middleware, converting video request fields.
- Add a `VideoWriter` that assembles the streaming response (parallel to
  `ImageWriter`).

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
At `cmd.go:886`, replace the `imagegen.RunCLI` call with a unified dispatch:

```go
if diffgen.IsDiffModel(name) {
    return diffgen.RunCLI(cmd, name, opts.Prompt, interactive, opts.KeepAlive)
}
```

The runner auto-detects image vs video mode from the model's capabilities.

#### 6.2 `x/diffgen/cli.go`
Mirror `x/imagegen/cli.go` (which it replaces). Differences:
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

### Phase 7: MLX removal and migration cleanup

**Goal:** All MLX code, build wiring, and references are removed. The codebase
is clean.

#### 7.1 Delete MLX packages
- `x/imagegen/` (entire directory — replaced by `x/diffgen/`)
- `x/imagegen/mlx/` (CGO bridge)
- `x/imagegen/models/flux2/`, `x/imagegen/models/zimage/` (MLX model impls)
- `x/mlxrunner/` (entire directory — MLX-based LLM runner)
- `x/models/` MLX-dependent model implementations (qwen3_5, qwen3_5_moe, etc.
  that import `x/mlxrunner/`) — these are MLX text-gen models and are removed
  with the MLX runner. Their GGUF equivalents run via llama.cpp.
- `x/internal/mlxthread/` (MLX thread affinity)
- `cmake/mlx/` and `cmake/vendor-mlx-c-headers.cmake`

#### 7.2 Remove MLX build wiring
- `cmake/local.cmake`: remove `ollama_add_mlx_build()`, `OLLAMA_MLX_BACKENDS`,
  MLX source fetching, `ollama-mlx-generate-wrappers`, `ollama-mlx-backends`.
- Delete `MLX_VERSION` and `MLX_C_VERSION` files.
- `Dockerfile`: remove MLX build dependencies.

#### 7.3 Remove MLX references in Go code
- `server/sched.go`: remove `mlxrunner` import (line 27), `IsMLX()` checks
  (lines 531, 1443), `mlxrunner.NewClient` branch (line 597).
- `server/routes.go`: remove `IsMLX()` checks (lines 567, 1519, 2363, 2701).
- `server/images.go`: remove `IsMLX()` method (line 84) and its callers.
- `x/quant/`, `x/safetensors/`, `x/create/`: update any MLX-dependent code to
  be backend-agnostic or remove if solely MLX-serving.
- `cmd/cmd.go`: remove `imagegen` import, `--imagegen` flag (line 2371),
  `use_imagegen_runner` option handling (line 222).

#### 7.4 Update documentation
- `AGENTS.md`: update architecture section to reflect two-backend model
  (llama.cpp for text, SD.cpp for image+video).
- `x/imagegen/README.md` → `x/diffgen/README.md`: rewrite for SD.cpp backend.
- `docs/development.md`: update build instructions (no MLX, add SD.cpp).

**Deliverable:** `go build ./...` and `cmake --build build` succeed with zero
MLX references. `grep -ri mlx .` returns nothing in source.

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
  end-to-end generation with small models (SD1.5-turbo for image, WAN 1.3B for
  video). Gated behind `OLLAMA_TEST_DIFF_MODEL`.
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

### Deleted files (MLX removal)
| Path | Reason |
|------|--------|
| `MLX_VERSION` | MLX pin removed |
| `MLX_C_VERSION` | MLX-C pin removed |
| `cmake/mlx/` (entire dir) | MLX CMake subproject |
| `cmake/vendor-mlx-c-headers.cmake` | MLX header vendoring |
| `x/imagegen/` (entire dir) | Replaced by `x/diffgen/` |
| `x/mlxrunner/` (entire dir) | MLX-based LLM runner removed |
| `x/internal/mlxthread/` (entire dir) | MLX thread affinity |
| `x/models/` MLX-dependent impls | MLX text-gen model implementations |

### Modified files
| Path | Change |
|------|--------|
| `CMakeLists.txt` | Include `cmake/sdcpp` if SD.cpp backends requested |
| `cmake/local.cmake` | Remove MLX build; add `ollama_add_sdcpp_build()`, `OLLAMA_SDCPP_BACKENDS` |
| `server/sched.go` | Remove mlxrunner/imagegen; dispatch `"image"`/`"video"` to `diffgen.NewServer` |
| `server/routes.go` | Remove `IsMLX()` checks; register `/v1/video/generations`, `/v1/video/edits` |
| `server/images.go` | Remove `IsMLX()` method |
| `server/create.go` | SD.cpp model import path (detect + component blobs) |
| `api/types.go` | Add video fields to `GenerateRequest`/`GenerateResponse` |
| `openai/openai.go` | Add `VideoGenerationRequest`/`VideoEditRequest` types + converters |
| `middleware/openai.go` | Add `VideoGenerationsMiddleware`/`VideoEditsMiddleware` + `VideoWriter` |
| `cmd/cmd.go` | Replace `imagegen.RunCLI` with `diffgen.RunCLI`; remove `--imagegen` flag |
| `Dockerfile` | Remove MLX deps; add SD.cpp build deps |
| `AGENTS.md` | Update architecture section (two-backend model) |

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **ggml version skew** between SD.cpp's vendored ggml and llama.cpp's | High | Build conflicts / symbol clashes | Build SD.cpp as a shared lib with hidden visibility (only `SD_API` exported). Verify no `ggml_*` symbols leak with `nm`/`dumpbin`. Phase 0 validation gate. |
| **WAN VAE CUDA/CPU only** (Metal/Vulkan unsupported) | High | macOS Metal / Vulkan users get slow video | Default to `--vae-on-cpu` on non-CUDA. Warn users. Track upstream for Metal VAE. Recommend CUDA for video. |
| **14B model VRAM** exceeds typical consumer GPUs | High | OOM on load | Use SD.cpp `max_vram` + `stream_layers` offload. Default to 1.3B models in docs. Recommend Q8_0 GGUF quantization. |
| **Video container encoding** adds heavy Go deps (ffmpeg) | Medium | Binary bloat / cross-compilation pain | Phase 2 streams PNG frames (no container dep). Phase 6 adds WebM via optional cgo ffmpeg or pure-Go muxer behind a build tag. |
| **Synchronous generate calls block** the runner thread | Medium | Can't handle concurrent requests | Runner serializes per-model (existing `imageGenMu` pattern → `diffGenMu`). Scheduler serializes per runnerRef. Acceptable for phase 1. |
| **SD.cpp API instability** (README: "API may change frequently") | Medium | Binding breakage on version bump | Pin a specific commit. Wrap all C calls in a thin Go interface so binding changes are localized to `x/sdcpp/`. |
| **MLX removal breaks safetensors LLM models** | Medium | Users of experimental safetensors LLMs lose support | These were experimental. Document removal. GGUF conversion path remains via llama.cpp. Out of scope for this plan. |
| **ggml shared-lib symbol hiding** not sufficient on Windows | Low | DLL load failures | On Windows, SD.cpp uses `__declspec(dllexport)` for `SD_API` only; ensure the build sets `SD_BUILD_DLL` correctly. Test on Windows early in phase 0. |
| **WebM support not compiled** in SD.cpp build | Low | `output_format: "webm"` returns 400 | Detect at context creation. Fall back to PNG frame stream. |

---

## 7. Phased delivery schedule

| Phase | Deliverable | Dependencies | Duration |
|-------|------------|--------------|----------|
| **0** | SD.cpp builds via CMake (all backends); MLX build wiring removed | None | 3 weeks |
| **1** | CGO bridge package `x/sdcpp/` compiles | Phase 0 | 2 weeks |
| **2** | Unified runner (`x/diffgen/`) generates image + video (manual model loading) | Phase 1 | 3 weeks |
| **3** | Scheduler dispatch + `ollama run` end-to-end | Phase 2 | 1 week |
| **4** | HTTP API endpoints (image preserved + video new) + streaming | Phase 3 | 2 weeks |
| **5** | Model import (`ollama create`) + manifest | Phase 3 | 2 weeks |
| **6** | CLI UX (progress bars, img2img/I2V, output formats) | Phase 4, 5 | 1 week |
| **7** | MLX removal + migration cleanup | Phase 2, 6 | 2 weeks |
| **8** | Multi-backend (Metal/Vulkan), VRAM offload, tests | Phase 7 | 3 weeks |
| | | **Total** | **~19 weeks (~4-5 months)** |

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
- **Safetensors LLM text generation.** The MLX runner supported experimental
  safetensors LLM models. These are removed with MLX. If needed, convert to GGUF
  and run via llama.cpp.

---

## 9. Key code references

| Concept | Location |
|---------|----------|
| Runner subprocess (llm.LlamaServer) pattern | `x/imagegen/server.go:35-154` (to be replaced) |
| Runner entry point | `x/imagegen/runner.go:22-115` (to be replaced) |
| Image completion streaming | `x/imagegen/imagegen.go:64-132` (to be replaced) |
| Scheduler dispatch (image/mlx) | `server/sched.go:592-599` (to be modified) |
| Scheduler MLX branch | `server/sched.go:531, 597` (to be removed) |
| Runner reload check | `server/sched.go:1393-1449` (to be modified) |
| OpenAI image middleware | `middleware/openai.go:601-680` (image: preserved) |
| OpenAI image types | `openai/openai.go:789-844` (image: preserved) |
| API types (Width/Height/Steps/Image) | `api/types.go:131-143`, `api/types.go:946` |
| Routes registration | `server/routes.go:1916-1917` |
| Model capabilities | `types/model/config.go:4-28` |
| Safetensors model import | `server/create.go:517-519` (to be extended) |
| Manifest + blob storage | `x/imagegen/manifest/manifest.go` (pattern reference) |
| Model type detection | `x/imagegen/memory.go:53-80` (to be replaced) |
| CLI dispatch | `cmd/cmd.go:886` (to be modified) |
| CLI image-gen flow | `x/imagegen/cli.go:82-194` (to be replaced) |
| Progress bar UX | `x/imagegen/cli.go:146-163` (pattern reference) |
| MLX CGO bridge (pattern, to be deleted) | `x/imagegen/mlx/mlx.go:1-46` |
| MLX CMake (pattern, to be deleted) | `cmake/local.cmake:142-195`, `x/imagegen/mlx/CMakeLists.txt` |
| llama.cpp backend build (pattern) | `cmake/local.cmake:363-450` (`ollama_add_llama_server_build`) |
| SD.cpp C API (image + video) | `include/stable-diffusion.h` (`generate_image`, `generate_video`, `sd_img_gen_params_t`, `sd_vid_gen_params_t`) |
| SD.cpp WAN docs | `docs/wan.md` (leejet/stable-diffusion.cpp) |

---

## 10. Summary

This plan consolidates all diffusion-based media generation (image **and** video)
onto a single native backend, stable-diffusion.cpp, removing the existing MLX
image-gen stack entirely. The resulting architecture is clean and uniform across
all platforms:

- **llama.cpp** → text generation (CUDA, Metal, Vulkan, ROCm, CPU).
- **stable-diffusion.cpp** → image + video generation (CPU, CUDA, Metal, Vulkan,
  OpenCL, SYCL).

SD.cpp is a superset of MLX's image capabilities (supporting all major image
model families: SD, SDXL, SD3, FLUX, Qwen-Image, Z-Image, etc.) **plus** native
video (WAN 2.1/2.2, LTX-2.3, LingBot-Video). It runs on every platform Ollama
targets, including macOS Metal (replacing MLX's macOS-only advantage) and adds
Vulkan/OpenCL/SYCL coverage MLX never had.

The hard work is concentrated in four areas:
1. **Build integration + ggml coexistence** — getting SD.cpp to compile as an
   isolated shared library alongside llama.cpp without symbol clashes, across
   all backends.
2. **CGO binding** — wrapping the SD.cpp C API (`generate_image`,
   `generate_video`, callbacks, cancellation) in idiomatic Go.
3. **Unified diffgen runner** — a single subprocess runner handling both image
   and video modes, replacing the MLX-based imagegen runner.
4. **MLX removal** — deleting the MLX packages, CMake wiring, and all references,
   while migrating image generation to SD.cpp.

Everything else — scheduler dispatch, manifest storage, blob addressing, CLI
framework, progress streaming, the existing OpenAI image API — is adaptation of
existing, tested code paths. A focused effort reaches a functional PoC (CUDA +
CPU, image + 1.3B video, frame-stream output) in ~8 weeks, and a
production-quality multi-backend release with MLX fully removed in ~4-5 months.
