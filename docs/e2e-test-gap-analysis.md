# E2E Test Gap Analysis: SD.cpp / Diffgen Integration

**Project:** aiollama (Ollama fork)
**Scope:** Section 12.6 — End-to-end testing with real SD.cpp models
**Date:** 2026-07-17
**Status:** Diagnostic report — IMAGE E2E DONE, VIDEO E2E pending download

---

## 1. Executive summary

The E2E test **harness** is implemented and the **image E2E test now passes
with real model weights**. `TestDiffgenImageGeneration` and
`TestDiffgenImageGenerationProgress` both succeed against a real
FLUX.2-Klein-4B model (Q4_0 diffusion + Q2_K text encoder + VAE) on CPU,
producing a ~500-700 KB PNG in ~11 minutes with 26 streaming progress events.

The test harness was previously blocked by 9 concrete gaps. All have been
resolved for the image path: the sdcpp-tagged binary is built, `LD_LIBRARY_PATH`
is documented, model filenames are corrected, `model_index.json` fixtures exist,
the import helper is recursive, a dummy-weight fixture enables automated import
testing, and the runner's model-loading path is validated against real weights.

A critical **undocumented ABI bug** was discovered and fixed during this work:
the vendored header `x/sdcpp/include/stable-diffusion.h` was severely outdated
relative to the built `libstable-diffusion.so`, causing struct field offsets to
diverge (e.g. `backend`, `max_vram`, `flash_attn` written into wrong fields) and
`ListDevices()` to return garbage. The header was synced, the Go types/bridge
were adapted to the new struct layouts (new `sd_ctx_params_t`, `sd_sample_params_t`,
`sd_img_gen_params_t`, `sd_vid_gen_params_t`), and `high_noise_diffusion_model_path`
wiring was added for WAN 2.2 dual-stage models.

Video E2E remains pending: the WAN 2.2 dual-model download (~10 GB) has not been
attempted, and CPU video generation is impractically slow for CI (minutes per
frame on a 14B model). A GPU-backed CI runner is needed for practical video E2E.

---

## 2. Current state (what exists)

| Component | State | Location |
|-----------|-------|----------|
| `libstable-diffusion.so` (CPU backend) | Built & validated (56 MB) | `build/lib/ollama/sdcpp/cpu/` |
| `ollama` binary (sdcpp-tagged) | **DONE** — built & links libstable-diffusion (51 MB) | `./ollama` |
| `x/sdcpp` CGO bridge | Compiles with sdcpp tag + lib path; **header synced to built lib** | `x/sdcpp/sdcpp.go` |
| `x/diffgen` runner (sdcpp) | Compiles + 25 mock tests pass | `x/diffgen/` |
| `convertFromSDCpp` unit tests | **DONE** (9 tests, all pass) | `server/routes_create_test.go` |
| Integration test harness | **DONE** (import helpers recursive, 5 tests, skip cleanly) | `integration/diffgen_test.go` |
| ggml symbol isolation | Validated (0 leaked symbols) | §12.1 |
| WebM container encoding | Implemented (ffmpeg-based) | §12.5 |
| `model_index.json` fixtures | **DONE** — FLUX.2, WAN 2.2, and dummy | `integration/testdata/diffgen/` |
| Dummy-weight fixture for CI import test | **DONE** | `integration/testdata/diffgen/flux2-dummy/` |
| Real-model image E2E (FLUX.2-Klein-4B) | **DONE** — PASS (~11 min CPU, 26 progress events) | — |
| Real-model video E2E (WAN 2.2) | **PENDING** — wiring ready, download not done | — |

---

## 3. Gaps blocking real-model E2E

### Gap 1: The `ollama` binary is NOT compiled with the `sdcpp` tag — RESOLVED

The `ollama` binary is now built with the `sdcpp` tag and links
`libstable-diffusion.so` (verified with `ldd`). The diffgen runner and
scheduler dispatch path are active.

**Fix applied:**
```bash
CGO_LDFLAGS="-Lbuild/lib/ollama/sdcpp/cpu -lstable-diffusion -lstdc++ -ldl" \
LD_LIBRARY_PATH=build/lib/ollama/sdcpp/cpu \
go build -tags=sdcpp -o ollama .
```

---

### Gap 2: `LD_LIBRARY_PATH` not propagated to the integration test subprocess — DOCUMENTED

The diffgen test binary loads `libstable-diffusion.so` at process startup via
cgo direct-linking. The parent `ollama serve` process must have the lib path in
its environment so the runner subprocess (`ollama runner --diffgen-engine`)
inherits it.

**Fix (documented):** Before running integration tests:
```bash
export LD_LIBRARY_PATH="$PWD/build/lib/ollama/sdcpp/cpu:$LD_LIBRARY_PATH"
```
Or install the lib system-wide (`cp .../libstable-diffusion.so /usr/local/lib/ && ldconfig`).
For tests using `OLLAMA_TEST_EXISTING`, the pre-started server must have
`LD_LIBRARY_PATH` set.

---

### Gap 3: Model filenames in the plan (§12.6) are INCORRECT — RESOLVED

The plan §12.6 filenames have been corrected based on verified HuggingFace API
queries (2026-07-17):

| Component | Old (wrong) | Correct | Source |
|-----------|-------------|---------|--------|
| FLUX.2 diffusion_model | `flux-2-klein-4b-Q2_K.gguf` (404) | `flux-2-klein-4b-Q4_0.gguf` | `leejet/FLUX.2-klein-4B-GGUF` (Q4_0 smallest) |
| FLUX.2 vae | `flux2_ae.safetensors` (gated repo) | `flux2-vae.safetensors` | `Comfy-Org/vae-text-encorder-for-flux-klein-4b` (non-gated) |
| FLUX.2 text encoder | `qwen-3-4b-Q2_K.gguf` (wrong case) | `Qwen3-4B-Q2_K.gguf` | `unsloth/Qwen3-4B-GGUF` (component: `llm`, not `clip_l`) |
| WAN t5xxl | `umt5-xxl-encoder-Q2_K.gguf` (404) | `umt5-xxl-encoder-Q3_K_S.gguf` | `city96/umt5-xxl-encoder-gguf` (Q3_K_S smallest) |
| WAN vae | `wan_2.1_vae.safetensors` (wrong case) | `Wan2.1_VAE.safetensors` | `QuantStack/Wan2.2-T2V-A14B-GGUF/VAE/` |
| WAN Low/HighNoise | flat filenames | `LowNoise/...` and `HighNoise/...` subdirs | `QuantStack/Wan2.2-T2V-A14B-GGUF` |

**Key discovery:** FLUX.2-Klein-4B uses `--llm` (Qwen3-4B) as the text encoder,
**not** `--clip_l`. The `model_index.json` component name must be `llm`, and the
runner's `createSDContext` now maps it to `llm_path`.

---

### Gap 4: No test model downloaded or imported — RESOLVED (image only)

FLUX.2-Klein-4B model files are downloaded (~4.4 GB total) and the model is
imported via `TestDiffgenImageGeneration` using `OLLAMA_TEST_DIFF_MODEL_DIR`.
The test passes end-to-end. Video model (WAN 2.2, ~10 GB) is not yet downloaded.

---

### Gap 5: `importDiffModelFromDir` reads a flat directory only — RESOLVED

`importDiffModelFromDir` now uses recursive `filepath.WalkDir` traversal, so
repos with subdirectory layouts (e.g. WAN 2.2's `LowNoise/`, `HighNoise/`,
`VAE/`) can be imported from their original structure without manual flattening.

---

### Gap 6: No `model_index.json` test fixtures exist — RESOLVED

Fixtures created in `integration/testdata/diffgen/`:
- `flux2-klein-4b/model_index.json` — FluxPipeline, image, components map
- `wan2.2-t2v-a14b/model_index.json` — WanVideoPipeline, video, dual-stage
- `flux2-dummy/model_index.json` + dummy weight files (valid magic bytes) for
  automated import testing without real model downloads

---

### Gap 7: Runtime model loading path is untested — RESOLVED (image)

The runner's `createSDContext` → `sdcpp.NewContext` → `new_sd_ctx` path is now
validated with real FLUX.2-Klein-4B weights. The component name → ctx param
mapping works: `diffusion_model` → `diffusion_model_path`, `vae` → `vae_path`,
`llm` → `llm_path`. The `high_noise_diffusion_model` component is wired to
`high_noise_diffusion_model_path` for WAN 2.2 dual-stage (untested with real
weights, but the mapping is in place).

**Critical ABI fix:** The vendored header was severely outdated, causing
struct field offsets to diverge. `sd_ctx_params_t` in the real header has
~20 new fields before `backend`/`max_vram`/`flash_attn` (including
`diffusion_model_path`, `high_noise_diffusion_model_path`, `llm_path`,
`embeddings`, `rng_type`, `prediction`, etc.), so the Go bridge was writing
`backend` and `max_vram` into wrong memory locations. The header was synced,
Go types adapted, and `sd_ctx_params_init()`/`sd_img_gen_params_init()`/
`sd_vid_gen_params_init()` are now called to zero-initialize structs before
populating fields.

---

### Gap 8: No dummy-weight fixture for automated import testing — RESOLVED

`integration/testdata/diffgen/flux2-dummy/` contains dummy weight files with
valid magic bytes (GGUF magic `0x47475546` + zeros, safetensors header). The
`TestDiffgenImportFromDirectory` test passes against this fixture in CI without
any real model download — it validates the full import → manifest → Show path.

---

### Gap 9: WAN VAE backend limitation on CPU — DOCUMENTED (not blocking for image)

The WAN VAE supports only CUDA and CPU. On CPU-only WSL, video VAE decode is
functional but very slow (minutes per frame for a 14B model). The
`WANVAEDeprecatedBackend` warning fires correctly on Metal/Vulkan. For image
generation (FLUX.2), the VAE runs on CPU without issues (tested, ~10s decode).
Video E2E on CPU is impractical for CI; a GPU runner is needed.

---

## 4. Environment inventory

| Item | Value |
|------|-------|
| Platform | WSL (Linux 5.15, x86-64) |
| Go version | 1.26.0 |
| GCC | 13.3.0 |
| ffmpeg | 6.1.1 (libvpx enabled) |
| `libstable-diffusion.so` | Built, CPU backend, 56 MB |
| `huggingface-cli` | NOT installed |
| Network to HuggingFace | OK (HTTP 200) |
| `OLLAMA_MODELS` | Unset (defaults to `~/.ollama/models`) |
| `OLLAMA_TEST_DIFF_MODEL` | Unset |
| `OLLAMA_TEST_DIFF_MODEL_DIR` | Unset |
| `OLLAMA_TEST_DIFF_IMPORT_DIR` | Unset |
| GPU | None (CPU-only WSL) |

---

## 5. Action plan (prioritized)

| # | Action | Effort | Blocks | Depends on |
|---|--------|--------|--------|------------|
| 1 | Rebuild `ollama` with `-tags=sdcpp` + lib path | 5 min | All E2E | None |
| 2 | Set `LD_LIBRARY_PATH` for test environment | 1 min | All E2E | #1 |
| 3 | Correct model filenames in plan §12.6 | 15 min | Downloads | None |
| 4 | Create `model_index.json` fixtures for FLUX.2 + WAN | 15 min | Import | #3 |
| 5 | Download FLUX.2-Klein-4B model files (~5-6 GB) | 30 min | Image E2E | #3 |
| 6 | Run `TestDiffgenImportFromDirectory` with FLUX.2 dir | 2 min | Validation | #1, #4, #5 |
| 7 | Run `TestDiffgenImageGeneration` + `TestDiffgenImageGenerationProgress` | 5 min | Image validation | #6 |
| 8 | (If image passes) Download WAN 2.2 model files (~15-20 GB) | 1h+ | Video E2E | #7 |
| 9 | Create WAN `model_index.json` + flat dir | 15 min | Video import | #8 |
| 10 | Run `TestDiffgenVideoGeneration` + `TestDiffgenVideoAPI` | 10 min | Video validation | #9 |
| 11 | (Optional) Enhance `importDiffModelFromDir` for recursive subdirs | 15 min | If keeping HF layout | None |
| 12 | (Optional) Create dummy-weight fixture for CI | 30 min | Automated import test | None |
| 13 | (Optional) Install `huggingface-cli` for scripted downloads | 5 min | Easier downloads | None |

**Critical path:** #1 → #2 → #3 → #4 → #5 → #6 → #7 (image E2E, ~1h total)
**Video path:** adds #8 → #9 → #10 (~2h+ including download)

---

## 6. What "done" looks like

### Image E2E (minimum viable) — DONE
```
$ export OLLAMA_TEST_DIFF_MODEL=flux2-klein-4b
$ export OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/flux2-klein-dir
$ export OLLAMA_TEST_EXISTING=1
$ export LD_LIBRARY_PATH=$PWD/build/lib/ollama/sdcpp/cpu
$ go test -tags=integration -run TestDiffgenImageGeneration ./integration/... -v -timeout 40m
--- PASS: TestDiffgenImageGeneration (650.77s)

$ go test -tags=integration -run TestDiffgenImageGenerationProgress ./integration/... -v -timeout 40m
--- PASS: TestDiffgenImageGenerationProgress (759.20s)
```

### Video E2E (full) — PENDING (download + GPU runner needed)
```
$ export OLLAMA_TEST_DIFF_MODEL=wan2.2-t2v-a14b
$ export OLLAMA_TEST_DIFF_MODEL_DIR=/path/to/wan22-dir
$ go test -tags=integration -run 'TestDiffgenVideo' ./integration/...
--- PASS: TestDiffgenVideoGeneration (XXX.Xs)
--- PASS: TestDiffgenVideoAPI (XXX.Xs)
```

### Import path (automated, no download) — DONE
```
$ export OLLAMA_TEST_DIFF_IMPORT_DIR=integration/testdata/diffgen/flux2-dummy
$ go test -tags=integration -run TestDiffgenImportFromDirectory ./integration/...
--- PASS: TestDiffgenImportFromDirectory (0.03s)
```

---

## 7. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SD.cpp runner cannot load GGUF weights (component mapping bug) | Medium | High — blocks all E2E | Test with dummy weights first; check `loadModel` → `new_sd_ctx` param mapping |
| CPU video generation exceeds 20-min timeout | High | Medium — test fails, not a real bug | Increase timeout to 60 min for CPU; or use Wan2.1 1.3B (lighter) |
| `model_index.json` field names don't match SD.cpp expectations | Medium | High — ctx creation fails | Verify against `sd_ctx_params_t` in `include/stable-diffusion.h` |
| FLUX.2 `ae.safetensors` is the wrong VAE for Klein-4B | Low | Medium — VAE decode fails | Check SD.cpp `docs/flux2.md` for the correct VAE filename |
| Q4_0 quantization too slow/large for CPU CI | Medium | Low — test slow but passes | Accept longer timeout; Q4_0 is the smallest available for FLUX.2 |
