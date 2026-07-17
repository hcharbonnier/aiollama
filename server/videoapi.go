package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/videojobs"
	"github.com/ollama/ollama/types/model"
)

// openaiPollAfterMS is the response header that tells the OpenAI SDK how long
// to wait before polling GET /v1/videos/{id} again. The SDK reads this header
// (falling back to 1000ms) on create and retrieve responses while the job is
// non-terminal. A few seconds is appropriate: video generation is
// long-running, and the store updates progress asynchronously.
const openaiPollAfterMS = "openai-poll-after-ms"

// defaultPollIntervalMS is the value sent in the openai-poll-after-ms header
// for non-terminal jobs. Long enough to avoid hammering the server, short
// enough that a completed job is noticed promptly.
const defaultPollIntervalMS = 2000

// VideoCreateHandler handles POST /v1/videos. It accepts multipart/form-data
// with prompt (required), model, seconds, size, and optional input_reference,
// creates an async video job, and responds with the Video object (status
// "queued"). The client polls GET /v1/videos/{id} for completion and fetches
// the MP4 via GET /v1/videos/{id}/content.
func (s *Server) VideoCreateHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	// Parse multipart/form-data (the OpenAI SDK always sends multipart for
	// POST /v1/videos, even when no file is attached). Fall back to JSON if
	// the client didn't send a multipart body, for convenience. Both paths
	// are bounded by a MaxBytesReader to prevent memory-exhaustion DoS via
	// an oversized request body (e.g. a large data URL in input_reference).
	const maxVideoBodyBytes int64 = 64 << 20 // 64 MiB
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxVideoBodyBytes)

	var params videoCreateInput
	contentType := c.GetHeader("Content-Type")
	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := c.Request.ParseMultipartForm(32 << 20); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to parse multipart form: "+err.Error()))
			return
		}
		if err := params.fromForm(c.Request); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
	} else {
		if err := c.ShouldBindJSON(&params); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
	}

	if params.Prompt == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
		return
	}
	if len(params.Prompt) > 32000 {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt must be 32000 characters or less"))
		return
	}

	if params.Model == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
		return
	}
	if params.Seconds == "" {
		params.Seconds = openai.VideoDefaultSeconds
	} else if !openai.VideoSecondsValues[params.Seconds] {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("seconds must be one of 4, 8, 12; got %q", params.Seconds)))
		return
	}
	if params.Size == "" {
		params.Size = openai.VideoDefaultSize
	} else if !openai.VideoSizeValues[params.Size] {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("size must be one of 720x1280, 1280x720, 1024x1792, 1792x1024; got %q", params.Size)))
		return
	}

	var initImage []byte
	if params.hasInputReference() {
		img, err := params.resolveInputReference(c.Request)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
		initImage = img
	}

	width, height, _ := strings.Cut(params.Size, "x")
	w, _ := strconv.Atoi(width)
	h, _ := strconv.Atoi(height)

	job, err := s.videoJobs.Create(videojobs.CreateParams{
		Model:     params.Model,
		Prompt:    params.Prompt,
		Seconds:   params.Seconds,
		Size:      params.Size,
		InitImage: initImage,
		Generate:  s.buildVideoGenerateFunc(params.Model, int32(w), int32(h), params.Seconds),
	})
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}

	c.Header(openaiPollAfterMS, strconv.Itoa(defaultPollIntervalMS))
	c.JSON(http.StatusOK, job.ToVideo())
}

// VideoRetrieveHandler handles GET /v1/videos/{video_id}. It returns the
// current state of a video job for client-side polling. The
// openai-poll-after-ms header is set on non-terminal responses so the SDK
// throttles its polling.
func (s *Server) VideoRetrieveHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	id := c.Param("video_id")
	job, err := s.videoJobs.Get(id)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video not found: "+id))
		return
	}

	v := job.ToVideo()
	if v.Status != openai.VideoStatusCompleted && v.Status != openai.VideoStatusFailed {
		c.Header(openaiPollAfterMS, strconv.Itoa(defaultPollIntervalMS))
	}
	c.JSON(http.StatusOK, v)
}

// VideoListHandler handles GET /v1/videos. It returns a cursor-paginated list
// of video jobs (most recent first by default).
func (s *Server) VideoListHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	after := c.Query("after")
	limit, _ := strconv.Atoi(c.Query("limit"))
	order := c.DefaultQuery("order", "desc")

	jobs, hasMore := s.videoJobs.List(after, limit, order)

	data := make([]openai.Video, 0, len(jobs))
	for _, j := range jobs {
		data = append(data, j.ToVideo())
	}

	resp := openai.VideoListResponse{
		Object:  openai.VideoObjectList,
		Data:    data,
		HasMore: hasMore,
	}
	if len(data) > 0 {
		resp.FirstID = data[0].ID
		resp.LastID = data[len(data)-1].ID
	}
	c.JSON(http.StatusOK, resp)
}

// VideoDeleteHandler handles DELETE /v1/videos/{video_id}. It cancels an
// in-flight job or removes a completed one, returning {deleted: true}.
func (s *Server) VideoDeleteHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	id := c.Param("video_id")
	deleted, err := s.videoJobs.Delete(id)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video not found: "+id))
		return
	}
	c.JSON(http.StatusOK, openai.VideoDeleteResponse{
		ID:      id,
		Deleted: deleted,
		Object:  openai.VideoObjectDeleted,
	})
}

// VideoContentHandler handles GET /v1/videos/{video_id}/content. It streams
// the binary MP4 for a completed job. The OpenAI SDK sets
// Accept: application/binary and reads the response as a binary stream.
// variant=video (default) returns the MP4; variant=thumbnail and
// variant=spritesheet are not yet implemented (501).
func (s *Server) VideoContentHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	id := c.Param("video_id")
	variant := c.DefaultQuery("variant", "video")
	if variant != "video" {
		c.AbortWithStatusJSON(http.StatusNotImplemented, openai.NewError(http.StatusNotImplemented, fmt.Sprintf("variant %q is not supported; only \"video\" is available", variant)))
		return
	}

	job, err := s.videoJobs.Get(id)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video not found: "+id))
		return
	}

	if job.Status() != openai.VideoStatusCompleted {
		v := job.ToVideo()
		status := v.Status
		if status == openai.VideoStatusFailed {
			c.AbortWithStatusJSON(http.StatusConflict, openai.NewError(http.StatusConflict, fmt.Sprintf("video generation failed: %s", errorMessage(v.Error))))
		} else {
			c.AbortWithStatusJSON(http.StatusConflict, openai.NewError(http.StatusConflict, fmt.Sprintf("video is not ready (status: %s)", status)))
		}
		return
	}

	content, contentType := job.Content()
	if len(content) == 0 {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video content is no longer available"))
		return
	}

	c.Header("Content-Type", contentType)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.mp4\"", id))
	c.Data(http.StatusOK, contentType, content)
}

// buildVideoGenerateFunc returns a videojobs.GenerateFunc that drives the
// diffgen runner via the scheduler. It captures the model name, requested
// dimensions, and the requested duration (seconds). The spec's `seconds`
// field (4/8/12) is translated into VideoFrames at 16 fps (WAN default) so
// the requested clip length drives the generated frame count; all other SD.cpp
// knobs (steps, cfg_scale, flow_shift, sampler) are left at zero/empty so the
// runner uses model defaults. This keeps the OpenAI endpoint spec-compliant
// (it only exposes prompt, seconds, size, input_reference) while the native
// /api/generate path remains fully parameterizable.
//
// The function collects each streamed frame (base64 PNG from the runner's
// CompletionResponse.Image) and decodes it to raw PNG bytes for the
// transcoder. Progress updates (Step/TotalSteps) are forwarded.
func (s *Server) buildVideoGenerateFunc(modelName string, width, height int32, seconds string) videojobs.GenerateFunc {
	// Derive frame count from the requested duration. The spec exposes only
	// seconds (4/8/12); we use 16 fps (the WAN default) so e.g. seconds=4 →
	// 64 frames. If seconds is empty/invalid, leave VideoFrames=0 (model
	// default).
	const videoFPS int32 = 16
	videoFrames := int32(0)
	if sec, err := strconv.Atoi(seconds); err == nil && sec > 0 {
		videoFrames = int32(sec) * videoFPS
	}

	return func(ctx context.Context, params videojobs.CreateParams, fn func(framePNG []byte, step, total int)) error {
		modelLoaded, err := GetModel(params.Model)
		if err != nil {
			return fmt.Errorf("load model: %w", err)
		}

		genCaps := []model.Capability{model.CapabilityVideo}
		if modelLoaded.CheckCapabilities(genCaps...) != nil {
			// Fall back to image capability if the model doesn't advertise
			// video (some imported models may only declare "image").
			genCaps = []model.Capability{model.CapabilityImage}
		}

		runner, m, _, err := s.scheduleRunner(ctx, params.Model, genCaps, nil, nil, nil)
		if err != nil {
			return fmt.Errorf("schedule runner: %w", err)
		}

		var media []llm.MediaData
		if len(params.InitImage) > 0 {
			media = append(media, llm.NewMediaData(0, params.InitImage))
		}

		// OutputFormat is left empty so the runner streams frames as PNG
		// (the frame-stream default), which the transcoder encodes to MP4.
		compReq := llm.CompletionRequest{
			Prompt:      params.Prompt,
			Width:       width,
			Height:      height,
			Media:       media,
			VideoFrames: videoFrames,
			FPS:         videoFPS,
			// Steps, CFGScale, FlowShift, Sampler: zero values → model
			// defaults (model_index.json / SD.cpp defaults).
		}

		err = runner.Completion(ctx, compReq, func(cr llm.CompletionResponse) {
			if cr.TotalSteps > 0 {
				fn(nil, cr.Step, cr.TotalSteps)
			}
			// NOTE: The runner emits frames as base64 PNG (via
			// EncodeImageBase64 → image.NewRGBA → png.Encode). The worker
			// decodes them back to RGB in the transcoder (pngToRGB). This
			// PNG encode/decode round-trip is a known perf cost; a future
			// optimization could add a GenerateFunc variant that streams
			// raw RGB directly (the runner holds raw sdcpp.Image.Data
			// before encoding to PNG), avoiding the round-trip entirely.
			if cr.Image != "" {
				// Decode the base64 PNG to raw bytes for the transcoder.
				if pngBytes, decErr := base64.StdEncoding.DecodeString(cr.Image); decErr == nil {
					fn(pngBytes, cr.Step, cr.TotalSteps)
				}
			}
		})

		if err != nil {
			s.sched.expireRunnersForRuntimeOOM(m, err)
			return err
		}
		return nil
	}
}

// videoCreateInput is the parsed POST /v1/videos request body. It supports
// both multipart/form-data (the spec-mandated content type, used by the SDK)
// and JSON (for convenience/curl).
type videoCreateInput struct {
	Prompt         string                    `json:"prompt" form:"prompt"`
	Model          string                    `json:"model,omitempty" form:"model,omitempty"`
	Seconds        string                    `json:"seconds,omitempty" form:"seconds,omitempty"`
	Size           string                    `json:"size,omitempty" form:"size,omitempty"`
	InputReference *openai.ImageInputReferenceParam `json:"input_reference,omitempty"`
	// inputReferenceFile is populated from a multipart file part named
	// "input_reference" (when the SDK uploads a file directly rather than
	// passing an image_url/file_id object).
	inputReferenceFile []byte
}

func (v *videoCreateInput) fromForm(r *http.Request) error {
	v.Prompt = r.FormValue("prompt")
	v.Model = r.FormValue("model")
	v.Seconds = r.FormValue("seconds")
	v.Size = r.FormValue("size")

	// input_reference can be a file part OR a JSON object in the
	// "input_reference" form field.
	if file, _, err := r.FormFile("input_reference"); err == nil {
		defer file.Close()
		data, err := io.ReadAll(file)
		if err != nil {
			return fmt.Errorf("read input_reference file: %w", err)
		}
		v.inputReferenceFile = data
		return nil
	}
	// No file part: check for a JSON object form field.
	if refStr := r.FormValue("input_reference"); refStr != "" {
		var ref openai.ImageInputReferenceParam
		if err := json.Unmarshal([]byte(refStr), &ref); err != nil {
			return fmt.Errorf("invalid input_reference: %w", err)
		}
		v.InputReference = &ref
	}
	return nil
}

func (v *videoCreateInput) hasInputReference() bool {
	return len(v.inputReferenceFile) > 0 ||
		(v.InputReference != nil && (v.InputReference.FileID != "" || v.InputReference.ImageURL != ""))
}

// resolveInputReference returns the raw image bytes for the input_reference,
// or an error. File parts and data URLs are supported; http(s) URLs and
// file_id are rejected (file_id requires a Files API upload store not
// implemented in v1).
func (v *videoCreateInput) resolveInputReference(r *http.Request) ([]byte, error) {
	if len(v.inputReferenceFile) > 0 {
		return v.inputReferenceFile, nil
	}
	if v.InputReference == nil {
		return nil, errors.New("input_reference is empty")
	}
	if v.InputReference.FileID != "" {
		return nil, openai.ErrVideoFileIDNotSupported
	}
	if v.InputReference.ImageURL != "" {
		img, err := openai.DecodeImageDataURL(v.InputReference.ImageURL)
		if err != nil {
			return nil, fmt.Errorf("input_reference.image_url: %w", err)
		}
		return img, nil
	}
	return nil, errors.New("input_reference must provide exactly one of image_url or file_id")
}

// errorMessage extracts the message from a *openai.VideoError, or returns a
// generic string.
func errorMessage(e *openai.VideoError) string {
	if e == nil {
		return "unknown error"
	}
	if e.Message != "" {
		return e.Message
	}
	return e.Code
}

// videoSourceInput is the parsed `video` reference for POST /v1/videos/edits
// and POST /v1/videos/extensions (spec §2.8/2.9). It can be:
//   - a multipart file part named "video" (uploaded MP4 bytes), OR
//   - a JSON object `{"id": "vid_..."}` referencing a previously-generated
//     completed job.
type videoSourceInput struct {
	// videoFile holds the uploaded MP4 bytes when the request sent a file
	// part. Empty when an id reference is used.
	videoFile []byte
	// videoID is the referenced job id when the request sent an {"id":...}
	// object. Empty when a file part is used.
	videoID string
}

// fromForm parses the `video` reference from a multipart form. The `video`
// field may be a file part (uploaded MP4) or a JSON object string. Returns
// the source input and the model/size fields (also parsed here so the edit/
// extend handlers share one parse pass).
func (v *videoSourceInput) fromForm(r *http.Request, fields map[string]string) error {
	if file, _, err := r.FormFile("video"); err == nil {
		defer file.Close()
		data, err := io.ReadAll(file)
		if err != nil {
			return fmt.Errorf("read video file: %w", err)
		}
		v.videoFile = data
		return nil
	}
	// No file part: check for a JSON object form field.
	if refStr, ok := fields["video"]; ok && refStr != "" {
		var ref openai.VideoReferenceParam
		if err := json.Unmarshal([]byte(refStr), &ref); err != nil {
			return fmt.Errorf("invalid video reference: %w", err)
		}
		v.videoID = ref.ID
	}
	return nil
}

// hasVideo reports whether a video reference was provided (file or id).
func (v *videoSourceInput) hasVideo() bool {
	return len(v.videoFile) > 0 || v.videoID != ""
}

// editExtendInput is the parsed, validated request for POST /v1/videos/edits
// and POST /v1/videos/extensions. It is produced by parseEditExtendRequest,
// which both handlers share to avoid duplicating ~80 lines of identical
// validation (nil-check, body limit, multipart parse, prompt/model/size
// validation, video source resolution, source-seconds lookup).
type editExtendInput struct {
	prompt        string
	model         string
	seconds       string
	size          string
	src           videoSourceInput
	sourceSeconds string
	width, height int
}

// parseEditExtendRequest performs the shared validation for the edit and
// extend handlers. It reads multipart/form-data, validates prompt/model/size,
// resolves the `video` reference, and looks up the source job's Seconds (for
// stitched-total reporting on extensions). secondsRequired controls whether
// `seconds` is mandatory (extensions) or optional with a default (edits);
// secondsValues selects the allowed set (VideoSecondsValues for edits,
// VideoExtensionSecondsValues for extensions). On any validation failure it
// aborts the gin context with a 400/503 and returns ok=false.
func (s *Server) parseEditExtendRequest(c *gin.Context, secondsRequired bool, secondsValues map[string]bool) (editExtendInput, bool) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return editExtendInput{}, false
	}

	const maxVideoBodyBytes int64 = 256 << 20 // 256 MiB (accommodate uploaded source MP4)
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxVideoBodyBytes)

	contentType := c.GetHeader("Content-Type")
	if !strings.HasPrefix(contentType, "multipart/form-data") {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "this endpoint requires multipart/form-data"))
		return editExtendInput{}, false
	}
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to parse multipart form: "+err.Error()))
		return editExtendInput{}, false
	}

	in := editExtendInput{
		prompt:  c.Request.FormValue("prompt"),
		model:   c.Request.FormValue("model"),
		seconds: c.Request.FormValue("seconds"),
		size:    c.Request.FormValue("size"),
	}

	if in.prompt == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
		return editExtendInput{}, false
	}
	if len(in.prompt) > 32000 {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt must be 32000 characters or less"))
		return editExtendInput{}, false
	}
	if in.model == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
		return editExtendInput{}, false
	}
	if secondsRequired {
		if in.seconds == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "seconds is required for extensions (4, 8, 12, 16, or 20)"))
			return editExtendInput{}, false
		}
	} else {
		if in.seconds == "" {
			in.seconds = openai.VideoDefaultSeconds
		}
	}
	if in.seconds != "" && !secondsValues[in.seconds] {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("seconds must be one of the allowed values; got %q", in.seconds)))
		return editExtendInput{}, false
	}
	if in.size == "" {
		in.size = openai.VideoDefaultSize
	} else if !openai.VideoSizeValues[in.size] {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("size must be one of 720x1280, 1280x720, 1024x1792, 1792x1024; got %q", in.size)))
		return editExtendInput{}, false
	}

	fields := map[string]string{
		"video": c.Request.FormValue("video"),
	}
	if err := in.src.fromForm(c.Request, fields); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return editExtendInput{}, false
	}
	if !in.src.hasVideo() {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "video is required (a file part or a {id} object)"))
		return editExtendInput{}, false
	}

	// Look up the source job's Seconds (for the remixed job's reporting)
	// when the source is a {id} reference.
	if in.src.videoID != "" {
		if job, err := s.videoJobs.Get(in.src.videoID); err == nil {
			in.sourceSeconds = job.ToVideo().Seconds
		}
	}

	width, height, _ := strings.Cut(in.size, "x")
	in.width, _ = strconv.Atoi(width)
	in.height, _ = strconv.Atoi(height)

	return in, true
}

// VideoEditHandler handles POST /v1/videos/edits. It accepts multipart/form-data
// with prompt (required) and video (required: a file part or a {id} object
// referencing a previously-generated completed video). The edit re-renders a
// new video from the source's first frame as an I2V init image with the new
// prompt. The result is the new generation (the source is not concatenated).
// Video.remixed_from_video_id is set when the source was a {id} reference.
//
// Spec: https://developers.openai.com/api/reference/resources/videos/edit
func (s *Server) VideoEditHandler(c *gin.Context) {
	in, ok := s.parseEditExtendRequest(c, false, openai.VideoSecondsValues)
	if !ok {
		return
	}

	job, err := s.videoJobs.Create(videojobs.CreateParams{
		Model:         in.model,
		Prompt:        in.prompt,
		Seconds:       in.seconds,
		Size:          in.size,
		RemixedFromID: in.src.videoID,
		SourceVideoID: in.src.videoID,
		SourceVideo:   in.src.videoFile,
		Extend:        false,
		SourceSeconds: in.sourceSeconds,
		Generate:      s.buildVideoGenerateFunc(in.model, int32(in.width), int32(in.height), in.seconds),
	})
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}

	c.Header(openaiPollAfterMS, strconv.Itoa(defaultPollIntervalMS))
	c.JSON(http.StatusOK, job.ToVideo())
}

// VideoExtendHandler handles POST /v1/videos/extensions. It accepts
// multipart/form-data with prompt (required), seconds (required, "4"-"20"),
// and video (required: file part or {id} object). The extension continues the
// source scene from its LAST frame as an I2V init image, then concatenates the
// source + the generated extension into a single clip. The response
// Video.seconds is the stitched total (source + requested).
//
// Spec: https://developers.openai.com/api/reference/resources/videos/extend
func (s *Server) VideoExtendHandler(c *gin.Context) {
	in, ok := s.parseEditExtendRequest(c, true, openai.VideoExtensionSecondsValues)
	if !ok {
		return
	}

	job, err := s.videoJobs.Create(videojobs.CreateParams{
		Model:         in.model,
		Prompt:        in.prompt,
		Seconds:       in.seconds,
		Size:          in.size,
		RemixedFromID: in.src.videoID,
		SourceVideoID: in.src.videoID,
		SourceVideo:   in.src.videoFile,
		Extend:        true,
		SourceSeconds: in.sourceSeconds,
		Generate:      s.buildVideoGenerateFunc(in.model, int32(in.width), int32(in.height), in.seconds),
	})
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
		return
	}

	c.Header(openaiPollAfterMS, strconv.Itoa(defaultPollIntervalMS))
	c.JSON(http.StatusOK, job.ToVideo())
}
