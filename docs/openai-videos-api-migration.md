# Migration to the Official OpenAI Videos API (Sora)

**Project:** aiollama (Ollama fork)
**Goal:** Make the OpenAI-compatible video layer **100% conformant** with the
official OpenAI Videos API (Sora) as specified by the OpenAI Python SDK and the
API reference at `https://developers.openai.com/api/reference/resources/videos/`.
**Author:** Engineering analysis
**Date:** 2026-07-17
**Status:** Phase 1 implemented and tested; Phase 2 (edits/extensions) deferred
**Replaces:** the non-spec-compliant `/v1/video/generations` + `/v1/video/edits`
endpoints documented in `docs/video-generation-implementation-plan.md` §4.3–4.4.

---

## 1. Background: the gap

The current OpenAI-compatible video layer was built on the **incorrect**
assumption that "OpenAI has no standardized video API as of 2026"
(`openai/openai.go:932-933`, `server/routes.go:1918`). This is false. OpenAI
ships a fully specified, asynchronous Videos API (Sora) with a Python SDK
(`openai.videos`), a REST surface, and Azure parity.

### What we built (non-conformant)

| Path | Real OpenAI path | Problem |
|------|------------------|---------|
| `POST /v1/video/generations` | `POST /v1/videos` | Wrong path (plural, no `/generations` suffix) |
| `POST /v1/video/edits` | `POST /v1/videos/edits` | Wrong path (plural) |
| — | `GET /v1/videos/{id}` | Missing (poll/retrieve) |
| — | `GET /v1/videos` | Missing (list) |
| — | `DELETE /v1/videos/{id}` | Missing (delete) |
| — | `GET /v1/videos/{id}/content` | Missing (binary download) |
| — | `POST /v1/videos/extensions` | Missing (deferred to Phase 2) |

Beyond the paths, the **request and response shapes are wrong**:

- **Request**: we accept JSON with `video_frames`, `fps`, `steps`, `cfg_scale`,
  `flow_shift`, `sampler`, `output_format`, `seed`, `negative_prompt`. The real
  API takes `multipart/form-data` with `prompt` (required), `model`,
  `seconds` (`"4"`|`"8"`|`"12"`), `size` (`"720x1280"`|…), and
  `input_reference` (`{file_id}` | `{image_url}`). None of our extra fields are
  in the spec; they are local SD.cpp knobs, not OpenAI parameters.
- **Response**: we return a **synchronous** `{"created":…, "data":[{"b64_json":…,"format":…}]}`.
  The real API returns an asynchronous **`Video` job object** (`{id, status,
  created_at, …}`); the MP4 is fetched separately via `GET /videos/{id}/content`
  as a **binary stream**.
- **Semantics**: the real API is a **job + poll + download** model; we are
  synchronous. OpenAI SDK clients (`client.videos.create(...).poll()` /
  `.download_content()`) will not work against us today.

### What is correct and stays

- The **native** `/api/generate` video path (`server/routes.go:427-430` →
  `handleImageGenerate`, `x/diffgen/server.go`, `x/diffgen/runner.go`) is the
  real generation engine and is **unchanged**. The CLI (`x/diffgen/cli.go`,
  `cmd/cmd.go:882-900`) uses `/api/generate` exclusively and is fully decoupled
  from the OpenAI endpoints.
- The image OpenAI layer (`/v1/images/generations`, `/v1/images/edits`) is
  conformant and untouched.
- The SD.cpp runner, WebM encoding (`x/diffgen/video.go:EncodeWebM`), VRAM
  estimation, and model import are all reusable as-is.

---

## 2. Authoritative OpenAI Videos API contract

Source: `openai-python` SDK source (`src/openai/resources/videos.py`,
`src/openai/types/video*.py`) + API reference. All SDK paths are prefixed with
`/v1` in the HTTP request.

### 2.1 Endpoints

| Method | Path | SDK method | Content-Type | Returns |
|--------|------|------------|--------------|---------|
| POST | `/v1/videos` | `videos.create` | `multipart/form-data` | `Video` (200) |
| GET | `/v1/videos/{video_id}` | `videos.retrieve` | — | `Video` (200) |
| GET | `/v1/videos` | `videos.list` | — | cursor list (200) |
| DELETE | `/v1/videos/{video_id}` | `videos.delete` | — | `VideoDeleteResponse` (200) |
| GET | `/v1/videos/{video_id}/content` | `videos.download_content` | `Accept: application/binary` | binary stream (200) |
| POST | `/v1/videos/edits` | `videos.edit` | `multipart/form-data` | `Video` (200) |
| POST | `/v1/videos/extensions` | `videos.extend` | `multipart/form-data` | `Video` (200) |

### 2.2 `Video` object (response, exact fields)

| field | type | always? |
|-------|------|---------|
| `id` | string | yes |
| `object` | literal `"video"` | yes |
| `created_at` | int (unix seconds) | yes |
| `completed_at` | int (unix seconds) | optional |
| `expires_at` | int (unix seconds) | optional |
| `model` | `VideoModel` (string) | yes |
| `status` | `"queued"` \| `"in_progress"` \| `"completed"` \| `"failed"` | yes |
| `progress` | int (0–100) | yes |
| `prompt` | string | optional |
| `seconds` | string (`Union[str, VideoSeconds]`; stitched total for extensions) | yes |
| `size` | `VideoSize` (string) | yes |
| `remixed_from_video_id` | string | optional |
| `error` | `{code, message}` or null | optional (failed only) |

### 2.3 Enum values

- **`status`**: `queued`, `in_progress`, `completed`, `failed`.
- **`VideoModel`** (string union): `sora-2`, `sora-2-pro`, `sora-2-2025-10-06`,
  `sora-2-pro-2025-10-06`, `sora-2-2025-12-08` (also accepts arbitrary string).
- **`VideoSeconds`** (request, **string**): `"4"`, `"8"`, `"12"`. Default `"4"`.
- **`VideoSize`** (**string**): `720x1280`, `1280x720`, `1024x1792`,
  `1792x1024`. Default `720x1280`.

> Critical: `seconds` and `size` are **strings**, not numbers, in both request
> and response. `created_at`/`completed_at`/`expires_at`/`progress` are ints.

### 2.4 `POST /v1/videos` body

`multipart/form-data`:

| name | type | required | notes |
|------|------|----------|-------|
| `prompt` | string | required | 1 ≤ len ≤ 32000 |
| `input_reference` | file part **or** `{file_id}` / `{image_url}` object | optional | exactly one of `file_id`/`image_url` |
| `model` | `VideoModel` | optional | default `sora-2` |
| `seconds` | `VideoSeconds` (string) | optional | default `"4"` |
| `size` | `VideoSize` (string) | optional | default `720x1280` |

`input_reference.image_url`: fully qualified URL or **base64-encoded data URL**
(maxLength 20971520). When passed as a file part, the SDK uploads it and
substitutes a `file_id`.

### 2.5 `GET /v1/videos/{video_id}/content`

- Query param `variant`: `"video"` (default, MP4) | `"thumbnail"` |
  `"spritesheet"`.
- Request header: `Accept: application/binary` (SDK forces it).
- Response: raw binary (MP4 for `variant=video`); streamed, not JSON.

### 2.6 `GET /v1/videos` (list)

Query params (all optional): `after` (cursor id), `limit` (int, 0–100),
`order` (`"asc"`|`"desc"`).

Response (cursor envelope, 200):
```json
{
  "object": "list",
  "data": [ /* Video[] */ ],
  "first_id": "vid_...",
  "has_more": true|false,
  "last_id": "vid_..."
}
```

### 2.7 `DELETE /v1/videos/{video_id}`

Response (200): `{id, deleted: bool, object: "video.deleted"}`.

### 2.8 `POST /v1/videos/edits` body (Phase 2)

`multipart/form-data`: `prompt` (required), `video` (file part **or** `{id}`
object, required — reference to a completed video to edit).

### 2.9 `POST /v1/videos/extensions` body (Phase 2)

`multipart/form-data`: `prompt` (required), `seconds` (`VideoSeconds`,
**required**, `"4"`–`"20"`), `video` (file part or `{id}`, required). Response
`Video.seconds` is the **stitched total** duration.

### 2.10 Polling model (no webhooks/SSE)

There are **no** SSE/streaming events for video creation. Completion is via
client-side polling: the SDK reads the `openai-poll-after-ms` response header
(fallback 1000 ms) from `retrieve` and loops until `status ∈ {completed, failed}`.

> Implication: our `POST /v1/videos` should set `openai-poll-after-ms` on the
> create response and on `retrieve` responses while `status` is non-terminal.

---

## 3. Architectural approach

### 3.1 Async job model

The real API is asynchronous. Our SD.cpp runner is synchronous (blocking call,
minutes of runtime). The bridge is an **in-memory job store + worker**:

```
POST /v1/videos
  → parse multipart → validate → create Job{status:queued}
  → enqueue job  (respond 200 with Video{status:queued})
  → worker goroutine:
       run diffgen Completion (streaming, native /api path in-process)
       update Job.progress along the way (status:in_progress)
       encode MP4 via ffmpeg (frames → MP4)
       store MP4 bytes (or path) on Job
       status:completed  (or status:failed + error)

GET /v1/videos/{id}
  → read Job → return Video object (+ openai-poll-after-ms header)

GET /v1/videos/{id}/content?variant=video
  → if status!=completed → 409/404 per spec behavior
  → stream MP4 bytes (Content-Type: video/mp4, Accept: application/binary)
```

### 3.2 Why in-memory for v1

- No existing job/async infrastructure exists in the codebase (the closest is
  `blobDownloadManager sync.Map` in `server/download.go:39`, a refcounted
  cancellable-progress registry — a good template).
- Video jobs are large and ephemeral; persistence (disk/SQLite) adds complexity
  without near-term value. The `expires_at` field gives us a spec-compliant
  way to bound retention; an in-memory TTL + size cap is the v1 strategy.
- The job store is behind an interface so a SQLite/disk-backed implementation
  can drop in later without touching handlers.

### 3.3 SD.cpp knobs not in the OpenAI spec

The spec exposes only `prompt`, `model`, `seconds`, `size`, `input_reference`.
Our SD.cpp runner needs `steps`, `cfg_scale`, `flow_shift`, `fps`,
`negative_prompt`, `seed`, `sampler`, `video_frames`. These are **not** OpenAI
parameters. Resolution:

- **Derived from spec fields where possible**: `video_frames` and `fps` are
  derived from `seconds` (e.g. `frames = seconds × fps`, with model defaults
  from `model_index.json` providing `fps` and a default `flow_shift`).
- **Model defaults** (`x/diffgen/manifest` `model_index.json` `defaults`): the
  import-time `defaults` block already carries `width`, `height`,
  `video_frames`, `fps`, `flow_shift`, `steps`, `cfg_scale`. The job worker
  fills these in; the OpenAI request only overrides `size` and `seconds`.
- **No passthrough of non-spec fields on `/v1/videos`**: we do **not** accept
  `cfg_scale` etc. on the OpenAI endpoint. Callers needing them use the native
  `/api/generate` (documented). This keeps us 100% spec-compliant.
- `negative_prompt`, `seed`, `sampler`: sourced from model defaults only for
  `/v1/videos`; the native path remains fully parameterizable.

### 3.4 MP4 production

The runner produces raw RGB frames (`sdcpp.Image`). `x/diffgen/video.go` already
pipes frames through an external `ffmpeg` subprocess to produce WebM
(`EncodeWebM`, `video.go:138-263`). For the OpenAI endpoint we need **MP4**:

- Add `EncodeMP4(ctx, frames []sdcpp.Image, fps int) ([]byte, error)` in
  `x/diffgen/video.go`, mirroring `EncodeWebM` but with `-c:v libx264 -pix_fmt
  yuv420p -f mp4 -movflags +faststart`. ffmpeg is the same optional runtime
  dependency (looked up on `PATH` via the existing `lookupFFmpeg()`).
- **Fallback if ffmpeg absent**: the spec requires MP4. If ffmpeg is missing,
  `POST /v1/videos` returns `status:failed` with `error.code = "ffmpeg_required"`
  (or the job fails at completion with the same). We do **not** silently serve
  WebM or PNG frames on `/content` — that would violate the spec contract.

### 3.5 `input_reference` → I2V

`POST /v1/videos` with `input_reference` is image-to-video (our I2V / WAN TI2V).
The middleware:
- If `input_reference` is a file part → read bytes, decode to `api.ImageData`.
- If `input_reference.image_url` is a `data:` URL → base64-decode (reuse
  `openai.decodeImageURL`, `openai/openai.go:680-709`).
- If `input_reference.image_url` is an `http(s)://` URL → **not supported**
  (matching existing image middleware behavior); reject with a clear error.
- If `input_reference.file_id` → v1: reject (`file_id` requires a Files API
  upload store we don't have); future work.
- The decoded image becomes `req.Images[0]` on the internal `api.GenerateRequest`,
  same as the native I2V path.

---

## 4. Phase 1 — Core async endpoints (create/retrieve/list/delete/download_content)

**Goal**: A client using the official `openai` SDK
(`client.videos.create(...).poll().download_content()`) works end-to-end
against a local aiollama server for text-to-video and image-to-video.

### 4.1 New package: `server/videojobs/` (job store + worker)

In-memory, concurrency-safe, TTL-expiring job store. Interface-based for future
disk/SQLite backing.

```
server/videojobs/
  store.go          # JobStore interface + in-memory implementation
  job.go            # Job struct (spec Video fields + internal state)
  worker.go         # runs diffgen Completion + MP4 encode, updates Job
  store_test.go     # unit tests (create/retrieve/list/delete/expire/cancel)
  worker_test.go    # unit tests with a mock completion fn (no GPU)
```

**`Job` struct** (internal; the HTTP-facing shape is `openai.Video`):
```go
type Job struct {
    ID          string        // "vid_<rand>"
    Model       string
    Prompt      string
    Seconds     string        // "4"|"8"|"12"
    Size        string        // "720x1280"|...
    Status      string        // queued|in_progress|completed|failed
    Progress    int           // 0-100
    CreatedAt   int64         // unix seconds
    CompletedAt int64         // unix seconds, 0 until done
    ExpiresAt   int64         // unix seconds
    Error       *VideoError   // {code, message}, nil unless failed
    Content     []byte        // MP4 bytes (completed only)
    ContentType string        // "video/mp4"
    cancel      context.CancelFunc
    mu          sync.RWMutex
}
```

**`JobStore` interface**:
```go
type JobStore interface {
    Create(params CreateParams) *Job            // enqueue + spawn worker
    Get(id string) (*Job, bool)
    Delete(id string) bool
    List(after string, limit int, order string) ([]*Job, bool) // (items, hasMore)
    Close() // cancel all workers on shutdown
}
```

**Worker** (`worker.go`):
- On `Create`: build an in-process `api.GenerateRequest` from the OpenAI params
  + model defaults (§3.3), derive `video_frames`/`fps` from `seconds`, set
  `OutputFormat` so the runner produces frames (we re-encode to MP4 here, not
  WebM — see §4.4), invoke the **scheduler/runner directly** (not an HTTP
  loopback) to get frames, call `EncodeMP4`, store bytes, set `completed`.
- Progress: the diffgen `Server.Completion` streams `Step`/`Total` and
  `Frame`/`Frames`; map these to `Progress` (0–100) on the Job.
- Failure: capture the runner error → `Status:failed`, `Error{code,message}`.
- Cancellation: `DELETE /v1/videos/{id}` cancels the worker context (the runner
  already supports `sd_cancel_generation` via `r.Context().Done()`,
  `x/diffgen/runner.go`).

### 4.2 New types in `openai/openai.go`

Replace the non-conformant `VideoGenerationRequest`/`VideoEditRequest`/
`VideoGenerationResponse`/`VideoURLOrData` with spec-accurate types:

```go
// Video is the spec Video object.
type Video struct {
    ID                 string         `json:"id"`
    Object             string         `json:"object"`               // "video"
    CreatedAt          int64          `json:"created_at"`
    CompletedAt        int64          `json:"completed_at,omitempty"`
    ExpiresAt          int64          `json:"expires_at,omitempty"`
    Model              string         `json:"model"`
    Status             string         `json:"status"`               // queued|in_progress|completed|failed
    Progress           int            `json:"progress"`
    Prompt             string         `json:"prompt,omitempty"`
    Seconds            string         `json:"seconds"`
    Size               string         `json:"size"`
    RemixedFromVideoID string         `json:"remixed_from_video_id,omitempty"`
    Error              *VideoError    `json:"error,omitempty"`
}

type VideoError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

type VideoDeleteResponse struct {
    ID      string `json:"id"`
    Deleted bool   `json:"deleted"`
    Object  string `json:"object"` // "video.deleted"
}

type VideoListResponse struct {
    Object  string  `json:"object"`   // "list"
    Data    []Video `json:"data"`
    FirstID string  `json:"first_id"`
    HasMore bool    `json:"has_more"`
    LastID  string  `json:"last_id"`
}

// ImageInputReferenceParam — input_reference object shape.
type ImageInputReferenceParam struct {
    FileID   string `json:"file_id,omitempty"`
    ImageURL string `json:"image_url,omitempty"`
}
```

`FromVideoGenerationRequest`/`ToVideoGenerationResponse`/`FromVideoEditRequest`
are **removed** (they encoded the wrong schema). The native `/api/generate`
path does not use them; the CLI is unaffected.

### 4.3 New handlers in `server/`

New file `server/videoapi.go` (mirrors the layout of handlers in
`server/routes.go` but isolated for reviewability):

```go
func (s *Server) VideoCreateHandler(c *gin.Context)      // POST /v1/videos
func (s *Server) VideoRetrieveHandler(c *gin.Context)    // GET  /v1/videos/:id
func (s *Server) VideoListHandler(c *gin.Context)        // GET  /v1/videos
func (s *Server) VideoDeleteHandler(c *gin.Context)      // DELETE /v1/videos/:id
func (s *Server) VideoContentHandler(c *gin.Context)     // GET  /v1/videos/:id/content
```

- **`VideoCreateHandler`**: `c.Request.ParseMultipartForm` (pattern from
  `middleware/openai.go:860` `TranscriptionMiddleware`); read `prompt`, `model`
  (default `sora-2` → mapped to the loaded Ollama model name via the request's
  `model` field — see §4.5), `seconds`, `size`; handle `input_reference` (file
  part or `{image_url}` object). Build `CreateParams`, call `s.videoJobs.Create`,
  respond 200 with `openai.Video{status:"queued"}` + `openai-poll-after-ms`
  header.
- **`VideoRetrieveHandler`**: `s.videoJobs.Get(id)` → `openai.Video`; set
  `openai-poll-after-ms` while non-terminal; 404 if not found.
- **`VideoListHandler`**: parse `after`/`limit`/`order` query →
  `s.videoJobs.List` → `openai.VideoListResponse`.
- **`VideoDeleteHandler`**: `s.videoJobs.Delete(id)` → `openai.VideoDeleteResponse`
  (`deleted:true`) or 404.
- **`VideoContentHandler`**: require `status==completed` (else 409 with a
  JSON error); `c.DataFromReader(200, len(content), "video/mp4", bytes.NewReader(content), nil)`;
  honor `variant=thumbnail|spritesheet` as 501-not-implemented for v1 (or 404).
  Set `Content-Disposition: attachment; filename="<id>.mp4"` (optional, helps
  SDK `.to_file()`).

### 4.4 MP4 encoding — `x/diffgen/video.go`

Add `EncodeMP4` alongside `EncodeWebM`:
```go
func EncodeMP4(ctx context.Context, frames []sdcpp.Image, fps int) ([]byte, error)
```
- Same `lookupFFmpeg()` / size-cap / streaming-stdin / bounded-stdout pattern
  as `EncodeWebM` (`video.go:138-263`).
- ffmpeg args: `-f rawvideo -pix_fmt rgb24 -s WxH -r fps -i - -an -c:v libx264
  -pix_fmt yuv420p -movflags +faststart -f mp4 -`.
- `SupportsMP4Encoding()` helper (mirrors `SupportsContainerEncoding()`).

The worker requests frames from the runner (not a WebM container), then calls
`EncodeMP4`. This is why the job worker sets `OutputFormat` to a frame-emitting
mode (or leaves it unset → frame stream is the default) rather than `"webm"`.

### 4.5 Model name mapping

The spec `model` field is `sora-2` / `sora-2-pro` / etc. We run local SD.cpp
models named e.g. `wan2.1-t2v-1.3b`. Two options (Phase 1 picks the explicit
one):

- **(Chosen)** `model` on `/v1/videos` is the **Ollama model name**
  (`wan2.1-t2v-1.3b`), exactly as on `/api/generate` and `/v1/images/generations`.
  The spec allows arbitrary strings for `VideoModel`, so this is conformant. The
  response `Video.model` echoes the requested name. This matches how
  `/v1/images/generations` already behaves (it takes the Ollama model name, not
  a literal `dall-e-3`). Documented as an aiollama extension within spec limits.
- (Alternative) a model alias map (`sora-2` → `wan2.1-t2v-1.3b`); rejected for
  v1 as it implies a registry we don't have and would surprise users who
  already use Ollama model names on the image endpoint.

### 4.6 Route registration — `server/routes.go`

**Remove** (`:1918-1920`):
```go
r.POST("/v1/video/generations", ..., middleware.VideoGenerationsMiddleware(), s.GenerateHandler)
r.POST("/v1/video/edits",       ..., middleware.VideoEditsMiddleware(),       s.GenerateHandler)
```

**Add**:
```go
r.POST(  "/v1/videos",            cloudPassthroughMiddleware(...), s.VideoCreateHandler)
r.GET(   "/v1/videos",            cloudPassthroughMiddleware(...), s.VideoListHandler)
r.GET(   "/v1/videos/:video_id",  cloudPassthroughMiddleware(...), s.VideoRetrieveHandler)
r.DELETE("/v1/videos/:video_id",  cloudPassthroughMiddleware(...), s.VideoDeleteHandler)
r.GET(   "/v1/videos/:video_id/content", cloudPassthroughMiddleware(...), s.VideoContentHandler)
```

Register the job store on `Server` (initialized in `NewServer`/`Serve`),
cancelled on shutdown.

### 4.7 Removal of non-conformant middleware/types

- Delete `VideoWriter`, `VideoGenerationsMiddleware`, `VideoEditsMiddleware`
  from `middleware/openai.go:682-811`.
- Delete `VideoGenerationRequest`, `VideoGenerationResponse`, `VideoURLOrData`,
  `VideoEditRequest`, `FromVideoGenerationRequest`, `ToVideoGenerationResponse`,
  `FromVideoEditRequest` from `openai/openai.go:931-1096`.
- The `ImageWriter`/`ImageGenerationsMiddleware`/`ImageEditsMiddleware` (image
  path) are **untouched**.

### 4.8 Tests

- **`server/videojobs/store_test.go`, `worker_test.go`**: unit tests with a
  mock completion function (no GPU, no SD.cpp). Cover create→in_progress→
  completed, failure path, cancellation, expiry, list pagination/order.
- **`server/videoapi_test.go`** (new): HTTP handler tests with a mock job store
  — multipart parsing, `input_reference` (file + data URL + http URL rejection
  + file_id rejection), defaults, 404s, poll-after header, content streaming,
  `variant` handling, delete.
- **`x/diffgen/video_test.go`**: add `TestEncodeMP4*` mirroring `TestEncodeWebM*`
  (ffmpeg-dependent, `t.Skip` if absent; validate MP4 magic / ftyp box).
- **`middleware/openai_test.go`**: **remove** `TestVideoGenerationsMiddleware`,
  `TestVideoWriterResponse`, `TestVideoEditsMiddleware` (`:1536-1887`) — the
  code they test is deleted.
- **`integration/diffgen_test.go`**: rewrite `TestDiffgenVideoAPI` (`:311-360`)
  to exercise the real flow: `POST /v1/videos` (multipart) → poll
  `GET /v1/videos/{id}` until `completed` → `GET /v1/videos/{id}/content` →
  assert MP4 bytes. Keep the `OLLAMA_TEST_DIFF_MODEL` gating. Add a test that
  the official `openai` Python SDK (if available in CI) can drive the flow
  (optional, behind a tag).

### 4.9 Doc/plan updates

- `docs/video-generation-implementation-plan.md` §4.3–4.4: replace the
  non-conformant endpoint spec with this conformant design; add a "Status:
  superseded by `docs/openai-videos-api-migration.md`" note on the old text.
- `AGENTS.md`: update the OpenAI-compatible API description to mention
  `/v1/videos` (async job model).
- `x/diffgen/README.md`: note the OpenAI endpoint layer and the ffmpeg-MP4
  requirement for `/v1/videos/*/content`.

### 4.10 Phase 1 deliverable checklist

- [x] `server/videojobs/` package (store + worker) with unit tests
- [x] `openai/openai.go` spec-accurate `Video`/`VideoError`/
      `VideoDeleteResponse`/`VideoListResponse`/`ImageInputReferenceParam`
- [x] `server/videoapi.go` handlers + `server/videoapi_test.go`
- [x] `x/diffgen/video.go` `EncodeMP4` + `SupportsMP4Encoding` + tests
  (and `server/videojobs/transcoder.go` for the server-side ffmpeg transcoder
  that does not depend on the sdcpp build tag)
- [x] `server/routes.go` route swap (remove old `/v1/video/*`, add new
      `/v1/videos/*`)
- [x] Delete non-conformant middleware + types + their tests
- [x] Rewrite `integration/diffgen_test.go::TestDiffgenVideoAPI`
- [x] Update `docs/video-generation-implementation-plan.md`, `AGENTS.md`,
      `x/diffgen/README.md`
- [x] `go build ./...`, `go test ./...` green; manual SDK smoke test (tests
      pass with sdcpp tag + libstable-diffusion linked)

---

## 5. Phase 2 — `edits` and `extensions` (deferred)

**Goal**: `POST /v1/videos/edits` and `POST /v1/videos/extensions` work,
reusing a previously-generated video as input.

### 5.1 Prerequisite: extract frames from a stored completed video

`/edits` and `/extensions` take a `video` reference (`{id}` of a completed job,
or a file part). For the `{id}` case we already have the MP4 bytes in the job
store. We need to **decode the MP4 back to frames** to feed the runner as
I2V/V2V input:

- Add `DecodeVideoFrames(ctx, mp4 []byte, maxFrames int) ([]sdcpp.Image, error)`
  in `x/diffgen/video.go`, using ffmpeg (`-i input.mp4 -f rawvideo -pix_fmt
  rgb24 -`). Reuses `lookupFFmpeg()`.
- For the file-part case (`video` as an uploaded MP4): same decode path on the
  uploaded bytes.

### 5.2 `POST /v1/videos/edits` handler

- Multipart: `prompt`, `video` (file or `{id}`).
- Resolve `video` to MP4 bytes (job store lookup by id, or uploaded file).
- Decode to frames; the first frame becomes `req.Images[0]` (I2V init frame)
  on the internal `api.GenerateRequest`. For true V2V (re-render with prompt),
  pass all frames as a control sequence if the model supports it (SD.cpp VACE /
  TI2V); otherwise treat as I2V from the first frame (documented limitation).
- Enqueue a job (same worker as create, but with the init image set).
- `Video.remixed_from_video_id` is set to the source video id when the source
  was a `{id}` reference.

### 5.3 `POST /v1/videos/extensions` handler

- Multipart: `prompt`, `seconds` (required, `"4"`–`"20"`), `video` (file or
  `{id}`).
- Resolve + decode source frames; the **last frame** of the source becomes the
  init frame for the extension segment (continue the scene).
- Generate the extension segment (same worker).
- The response `Video.seconds` is the **stitched total** (source + extension),
  per spec. The content (`/content`) is the stitched MP4 (concatenate source +
  extension via ffmpeg `concat` demuxer, or re-encode). For v2, simplest is to
  store the stitched MP4 on the new job.

### 5.4 Phase 2 deliverable checklist

- [ ] `x/diffgen/video.go` `DecodeVideoFrames` + tests
- [ ] `server/videoapi.go` `VideoEditHandler`, `VideoExtendHandler`
- [ ] `server/videojobs/worker.go` V2V/I2V-from-stored-video path
- [ ] Stitched-total `seconds` + concatenated content for extensions
- [ ] Handler + integration tests (mock + real model)
- [ ] Doc update (§5 of this report folded into the plan)

---

## 6. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **In-memory job store lost on restart** | Certain | In-flight jobs lost; `retrieve` 404s after restart | Spec allows `expires_at`; set a short TTL; document that v1 is in-memory. Future: disk/SQLite backing behind the interface. |
| **ffmpeg required for MP4** | Medium | `/content` cannot produce MP4 without ffmpeg | `POST` returns `queued`, job fails with `error.code="ffmpeg_required"`. Detect at create time if possible and fail fast. Document ffmpeg as required for the OpenAI video API (unlike the native path, where it's optional). |
| **Large MP4 in memory** | Medium | OOM on long videos | Size cap on stored content (reuse `maxWebMOutputBytes` pattern); reject jobs that exceed it at encode time; document the cap. Future: spill to a temp file and stream from disk in `VideoContentHandler`. |
| **SDK `seconds` enum ≠ model native fps** | Low | Frame count mismatch | Derive `video_frames = round(seconds × modelDefaultFps)`; clamp to model max. Documented derivation. |
| **Polling load** | Low | Many `retrieve` calls | `openai-poll-after-ms` header (set to e.g. 2000–5000 ms) throttles SDK polling; SDK honors it. |
| **`file_id` input_reference** | Low | 501 for v1 | Reject with a clear `error.code="file_id_not_supported"`; document. `image_url` data-URL covers the common case. |
| **Concurrent job limits** | Medium | Too many simultaneous video jobs OOM the runner | Job store enforces a max-concurrent-jobs semaphore (default 1, matching the existing `diffGenMu` per-runner serialization); excess `POST /v1/videos` returns `queued` and waits (spec-compatible) or 429 (documented). |
| **Backwards compatibility** | Medium | Existing callers of `/v1/video/generations` break | These callers were never spec-compliant and the endpoints were marked experimental. Announce removal in release notes. The native `/api/generate` video path is the stable, unchanged alternative. |

---

## 7. Out of scope (future work)

- **Disk/SQLite job store** for persistence across restarts (behind the
  `JobStore` interface).
- **`variant=thumbnail` / `variant=spritesheet`** on `/content` (v1 returns 501
  or 404 for these; only `variant=video` is implemented).
- **`POST /v1/videos/characters`** and **`GET /v1/videos/characters/{id}`**
  (character creation; Sora-specific, no SD.cpp equivalent).
- **`POST /v1/videos/{id}/remix`** (legacy/deprecated in the SDK; skipped).
- **`file_id`-based `input_reference`** (requires a Files API upload store).
- **Webhooks** (the spec mentions a generic webhooks surface; not
  video-specific and not needed for SDK polling compatibility).
- **Audio** (`sd_audio_t` from SD.cpp / Sora synced audio) — separate phase.

---

## 8. Implementation order (Phase 1)

1. `openai/openai.go` — add spec types (§4.2); leave old types in place
   temporarily to keep the build green.
2. `x/diffgen/video.go` — `EncodeMP4` + `SupportsMP4Encoding` + tests (§4.4).
3. `server/videojobs/` — store + worker + unit tests with mock completion (§4.1).
4. `server/videoapi.go` — handlers + HTTP tests with mock store (§4.3).
5. `server/routes.go` — wire new routes, keep old ones until tests migrate (§4.6).
6. Remove old middleware/types/tests (§4.7, §4.8).
7. Rewrite `integration/diffgen_test.go::TestDiffgenVideoAPI` (§4.8).
8. Docs + plan update (§4.9).
9. `go build ./...` + `go test ./...` green; manual SDK smoke test.

Each step compiles and tests independently; the old endpoints remain functional
until step 6, allowing incremental review.
