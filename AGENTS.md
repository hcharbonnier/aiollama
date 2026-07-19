# aiollama

aiollama is a fork of [Ollama](https://github.com/ollama/ollama) that extends
it with multimodal generation capabilities beyond text. Ollama is a Go
application that bundles native code (compiled with CGO) to run large language
models locally and exposes a REST API and CLI for pulling, creating, and
running models with optional GPU acceleration (Metal, CUDA, ROCm, Vulkan, MLX).

## Fork Goals

This fork adds the following capabilities on top of upstream Ollama:

- **Image generation/editing** via [stablediffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  (SD.cpp), integrated as a third native backend alongside llama.cpp and MLX.
- **Video generation** via SD.cpp (WAN, LTX, …) — SD.cpp is the only video
  backend.
- **Audio generation/editing** (planned for a later phase; backend not yet chosen).

The fork carries three native inference stacks, dispatched per model by the
scheduler (`server/sched.go`):

| Stack | Purpose | Platforms |
|-------|---------|-----------|
| llama.cpp | Text (LLM) generation from GGUF | Win, Linux, Mac |
| MLX | Image gen for natively-supported models (Z-Image, FLUX.2) + safetensors LLM text on macOS | Mac (primary), CUDA |
| stable-diffusion.cpp | Video (all models) + image gen for models MLX does not support + image/video on Linux/Windows | Win, Linux, Mac |

Routing uses `Config.ModelFormat`: `"gguf"`/`""` → llama.cpp, `"safetensors"`
→ MLX, `"sdcpp"` → SD.cpp. Follow the runner/subprocess architecture (see
`llm/`, `x/imagegen/`, and `x/diffgen/` for reference) and expose new
capabilities through the OpenAI-compatible API layer (`openai/`, `middleware/`)
as well as the CLI (`cmd/`).

## Building

build and test the app in WSL.

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

After changing native code, force a clean rebuild of the CGO payload with
`go clean -cache` to avoid stale data structures.

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.

## Code Style

- Go 1.26+ for all new code; follow `gofmt` and `gofumpt` (enforced by `.golangci.yaml`)
- Use tabs for Go indentation; follow standard `gofmt` formatting
- Do not add comments unless necessary; exported symbols follow the comment conventions disabled in staticcheck (`ST1020`–`ST1023`)
- Commit messages follow `<package>: <short description>` (see `CONTRIBUTING.md`), e.g. `llm/backend/mlx: support the llama architecture`
- Add dependencies sparingly and justify them in the PR description
- Avoid breaking backwards compatibility in the API (including the OpenAI-compatible API)

## Architecture

- `main.go` is the entry point; CLI commands live in `cmd/` (built on `spf13/cobra`)
- `server/` contains the HTTP server, routes (`routes.go`), model management, scheduling (`sched.go`), and the scheduler
- `llm/` manages the llama.cpp runner subprocess and native inference backends
- `discover/` performs GPU/CPU detection and selects acceleration libraries per platform (per-OS files: `*_darwin.go`, `*_linux.go`, `*_windows.go`)
- `ml/` provides the ML backend abstraction (`backend.go`) and backend implementations
- `model/` holds model parsers and template renderers (`parsers/`, `renderers/`)
- `api/` defines the public Go client and API types (`types.go`)
- `openai/` and `middleware/` provide OpenAI- and Anthropic-compatible API layers
- `fs/` handles GGML/GGUF file parsing and config; `envconfig/` centralizes `OLLAMA_*` environment configuration
- `server/videojobs/` implements the OpenAI Videos API (`/v1/videos`) async
  job store + worker (bridges the SDK's job/poll/download model with the
  synchronous diffgen runner; transcodes frames to MP4 via ffmpeg)
- `server/imageapi.go` implements the OpenAI Images API (`/v1/images/generations`,
  `/v1/images/edits`) as dedicated handlers driving the scheduler directly
  (n>1, multipart edits with mask→SD.cpp inpaint plumbing, output transcoding,
  `usage`); `server/imagefiles.go` is the TTL store behind `response_format=url`
- `x/` contains experimental subsystems:
  - `x/imagegen/` — MLX image generation (Z-Image, FLUX.2 on macOS, retained)
  - `x/diffgen/` — SD.cpp image + video generation runner (video + broad image coverage, all platforms)
  - `x/sdcpp/` — CGO bridge to the stable-diffusion.cpp C library
  - `x/mlxrunner/` — MLX safetensors LLM text runner (9 architectures on macOS, retained)
  - `x/create/`, `x/safetensors/` — model import utilities
- `integration/` holds end-to-end tests (disabled by default; enable with `-tags=integration`)
- Platform-specific files use Go build constraints (e.g. `*_windows.go`); respect this pattern for new platform code
- Native payloads are looked up in `build/lib/ollama`, `dist/<platform>/lib/ollama`, and standard install prefixes

## Testing

- Run unit tests: `go test ./...`
- Integration tests are gated behind build tags; run with `go test -tags=integration ./...` (some need `,models` and long timeouts)
- Integration tests expect a built `ollama` binary at the repo root; run `go build .` first
- On Windows, set `OLLAMA_HOST` for the managed integration server; on Unix a random port is used
- Set `OLLAMA_TEST_EXISTING` to target an already-running server; `OLLAMA_TEST_MODEL` to test a specific model
- Write table-driven tests (see `format/bytes_test.go` for the convention); test behavior, not implementation
- Test files use the `*_test.go` suffix and `package <pkg>` (not `_test`) unless black-box testing is intended

## Security

- Never commit API keys, secrets, or credentials
- Report security vulnerabilities privately to hello@ollama.com (do not open public issues; see `SECURITY.md`)
- Validate all inputs at API boundaries; use parameterized queries for SQLite access (`mattn/go-sqlite3`)
- Users should secure hosted Ollama instances and monitor for unusual activity

## Git

- You are responsible to add new commits when needed
- Create a new commit with a descriptive message