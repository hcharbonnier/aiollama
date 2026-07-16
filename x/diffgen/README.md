# diffgen

`x/diffgen` is the unified runner subprocess for image and video generation
via the [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(SD.cpp) native backend. It replaces the former MLX-based `x/imagegen` package
for all diffusion workloads.

## Architecture

The runner implements `llm.LlamaServer` and is spawned as a subprocess
(`ollama runner --diffgen-engine`), exposing a local HTTP server with
`/health` and `/completion` handlers that stream ndjson progress + results.
This mirrors the proven subprocess pattern from `x/imagegen/server.go`.

```
ollama run <model> "prompt"
  → scheduler (sched.go) dispatches to diffgen.NewServer when capability is
    "image" (sdcpp format) or "video"
  → diffgen.Server (server.go) spawns subprocess `ollama runner --diffgen-engine`
  → diffgen.Execute (runner.go) starts HTTP server in subprocess
  → Server.Completion (server.go) POSTs to child /completion
  → child handler dispatches to handleImageCompletion or handleVideoCompletion
    based on model capabilities (queried via sdcpp.SupportsVideoGeneration)
  → streams ndjson {step,total} then {image} or per-frame {frame,frames,image}
```

## Build tags

Files tagged `//go:build sdcpp` require `libstable-diffusion` to be linked
(via the `x/sdcpp` CGO bridge). Files without the tag (CLI, flags, detect,
memory, manifest) compile in all builds. When the `sdcpp` tag is absent,
`stub.go` provides stub `NewServer`/`Execute` that return a clear runtime
error.

## CLI usage

```
# Image generation
ollama run <image-model> "a cat on the moon"
ollama run <image-model> "edit this" -i photo.png

# Video generation
ollama run <video-model> "a lovely cat playing" --video-frames 33 --fps 16
ollama run <video-model> "animate this" --video-frames 49 -i photo.png

# Interactive REPL
ollama run <model>
>>> /set frames 49
>>> /set fps 24
>>> /set flow_shift 3.0
>>> a cat playing
```

### Flags

| Flag | Description |
|------|-------------|
| `--width` | Image/video width (default 1024) |
| `--height` | Image/video height (default 1024) |
| `--steps` | Denoising steps (0 = model default) |
| `--seed` | Random seed (0 = random) |
| `--negative` | Negative prompt |
| `--cfg-scale` | Classifier-free guidance scale |
| `--sampler` | Sampler name (e.g. euler) |
| `--output-format` | Output format (image: png; video: gif, webm, webp) |
| `--video-frames` | Number of video frames to generate |
| `--fps` | Output frame rate for video |
| `--flow-shift` | Flow shift for WAN video models |
| `--init-image` | Path to init image (img2img / I2V) |
| `--end-image` | Path to end frame image (FLF2V video) |

## Model manifest

SD.cpp reads whole checkpoint files (not per-tensor splits). Each component
(diffusion_model, vae, t5xxl, clip_vision) is stored as a single
content-addressed blob. See `manifest/manifest.go` for the component-file
manifest format and `memory.go` for model type detection (image vs video).

## Video output

Phase 1 streams PNG frames individually via ndjson (no container dependency).
The CLI assembles frames into a GIF (pure-Go `image/gif`) by default. WebM
container encoding is planned for a later phase behind a build tag.
