package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"strconv"
	"strings"
	"time"

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

// maxVideoPromptChars bounds the prompt length on all video endpoints
// (spec: 32000).
const maxVideoPromptChars = 32000

// validateVideoPrompt enforces the shared prompt rules for the video
// endpoints. On failure it aborts the gin context with a 400 and returns
// false.
func validateVideoPrompt(c *gin.Context, prompt string) bool {
	if prompt == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
		return false
	}
	if len(prompt) > maxVideoPromptChars {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("prompt must be %d characters or less", maxVideoPromptChars)))
		return false
	}
	return true
}

// videoSizeDimensions splits a validated "WxH" video size into width and
// height. Callers must have validated the size against VideoSizeValues (or
// inherited it from a validated job) first.
func videoSizeDimensions(size string) (int, int) {
	width, height, _ := strings.Cut(size, "x")
	w, _ := strconv.Atoi(width)
	h, _ := strconv.Atoi(height)
	return w, h
}

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

	if !validateVideoPrompt(c, params.Prompt) {
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

	w, h := videoSizeDimensions(params.Size)

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
// the binary content for a completed job. The OpenAI SDK sets
// Accept: application/binary and reads the response as a binary stream.
// variant=video (default) returns the MP4; variant=thumbnail returns the
// first frame as a PNG; variant=spritesheet returns a tiled frame grid PNG.
func (s *Server) VideoContentHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	id := c.Param("video_id")
	variant := c.DefaultQuery("variant", "video")
	if variant != "video" && variant != "thumbnail" && variant != "spritesheet" {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("variant must be one of video, thumbnail, spritesheet; got %q", variant)))
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

	switch variant {
	case "thumbnail", "spritesheet":
		tc := s.videoJobs.Transcoder()
		if tc == nil || !tc.Available() {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "ffmpeg is required on PATH to produce image variants"))
			return
		}
		// Variants are derived from the immutable job MP4; compute once and
		// cache on the job rather than re-running ffmpeg per download.
		png, err := job.CachedVariant(variant, func() ([]byte, error) {
			if variant == "thumbnail" {
				frames, _, err := tc.DecodeFrames(c.Request.Context(), content, 1)
				if err != nil {
					return nil, err
				}
				if len(frames) == 0 {
					return nil, errors.New("no frame extracted")
				}
				return frames[0], nil
			}
			return tc.Spritesheet(c.Request.Context(), content)
		})
		if err != nil || len(png) == 0 {
			if err == nil {
				err = errors.New("no frame extracted")
			}
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, fmt.Sprintf("%s extraction failed: %v", variant, err)))
			return
		}
		c.Header("Content-Type", "image/png")
		c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s_%s.png\"", id, variant))
		c.Data(http.StatusOK, "image/png", png)
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
	Prompt         string                           `json:"prompt" form:"prompt"`
	Model          string                           `json:"model,omitempty" form:"model,omitempty"`
	Seconds        string                           `json:"seconds,omitempty" form:"seconds,omitempty"`
	Size           string                           `json:"size,omitempty" form:"size,omitempty"`
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

	// input_reference: file part, JSON object form field, or SDK
	// bracket-notation fields (input_reference[image_url],
	// input_reference[file_id]).
	var ref openai.ImageInputReferenceParam
	var imageURL, fileID string
	data, err := parseMultipartRefField(r, "input_reference", &ref, map[string]*string{
		"image_url": &imageURL,
		"file_id":   &fileID,
	})
	if err != nil {
		return err
	}
	v.inputReferenceFile = data
	switch {
	case ref.ImageURL != "" || ref.FileID != "":
		v.InputReference = &ref
	case imageURL != "" || fileID != "":
		v.InputReference = &openai.ImageInputReferenceParam{ImageURL: imageURL, FileID: fileID}
	}
	return nil
}

// parseMultipartRefField parses a multipart "reference" field as emitted by
// the openai-python SDK, in one of three wire forms: a file part named
// <field> (returned as fileData), a JSON object string in the <field> form
// value (unmarshaled into jsonTarget), or bracket-notation subfields
// "<field>[sub]" (the SDK serializes nested dicts via qs.stringify_items,
// array_format="brackets"), each assigned to its target string. Precedence:
// file part, then JSON string, then bracket fields; empty bracket values
// never clobber.
func parseMultipartRefField(r *http.Request, field string, jsonTarget any, brackets map[string]*string) (fileData []byte, err error) {
	if file, _, err := r.FormFile(field); err == nil {
		defer file.Close()
		data, err := io.ReadAll(file)
		if err != nil {
			return nil, fmt.Errorf("read %s file: %w", field, err)
		}
		return data, nil
	}
	if refStr := r.FormValue(field); refStr != "" {
		if err := json.Unmarshal([]byte(refStr), jsonTarget); err != nil {
			return nil, fmt.Errorf("invalid %s: %w", field, err)
		}
		return nil, nil
	}
	for sub, dst := range brackets {
		if val := r.FormValue(field + "[" + sub + "]"); val != "" {
			*dst = val
		}
	}
	return nil, nil
}

func (v *videoCreateInput) hasInputReference() bool {
	return len(v.inputReferenceFile) > 0 ||
		(v.InputReference != nil && (v.InputReference.FileID != "" || v.InputReference.ImageURL != ""))
}

// resolveInputReference returns the raw image bytes for the input_reference.
// File parts, data URLs, and remote http(s) image URLs are supported;
// file_id is rejected (it requires a Files API upload store not implemented
// in v1).
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
		u := v.InputReference.ImageURL
		if strings.HasPrefix(u, "http://") || strings.HasPrefix(u, "https://") {
			return downloadRemoteImage(r.Context(), u)
		}
		img, err := openai.DecodeImageDataURL(u)
		if err != nil {
			return nil, fmt.Errorf("input_reference.image_url: %w", err)
		}
		return img, nil
	}
	return nil, errors.New("input_reference must provide exactly one of image_url or file_id")
}

// maxRemoteImageBytes bounds a downloaded input_reference image.
const maxRemoteImageBytes int64 = 25 << 20 // 25 MiB

// errBlockedRemoteHost is returned when an input_reference.image_url resolves
// to a non-public destination.
var errBlockedRemoteHost = errors.New("input_reference.image_url: host is not allowed (private, loopback, link-local, or otherwise non-public address)")

// blockedRemoteCIDRs lists special-use ranges that are not globally routable
// but are not covered by net.IP's IsPrivate/IsLoopback/etc. predicates:
// RFC 6598 shared address space (CGNAT — used by Tailscale tailnets),
// RFC 6890 protocol assignments, RFC 2544 benchmarking, and reserved space.
var blockedRemoteCIDRs = []netip.Prefix{
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("240.0.0.0/4"),
}

// isBlockedRemoteIP reports whether ip is a non-public destination
// (loopback, private RFC1918/RFC4193, CGNAT/shared, link-local, unspecified,
// multicast, or other special-use ranges). input_reference downloads are
// restricted to public IPs to prevent SSRF against internal services on
// deployments where the server is exposed.
func isBlockedRemoteIP(ip net.IP) bool {
	if ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() ||
		ip.IsMulticast() {
		return true
	}
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return true // unparseable: fail closed
	}
	addr = addr.Unmap()
	for _, p := range blockedRemoteCIDRs {
		if p.Contains(addr) {
			return true
		}
	}
	return false
}

// remoteImageClient is used for input_reference.image_url downloads. Its
// Transport pins each connection to a validated public IP: the DialContext
// resolves the host, rejects non-public destinations, and dials the resolved
// public IP directly. This both blocks SSRF to internal addresses and pins
// the IP at connect time (DNS rebinding between resolve and dial is moot).
// TLS ServerName still comes from the URL hostname, so HTTPS verification is
// unaffected. Redirects are safe: every new connection goes through the same
// dial validation.
var remoteImageClient = &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, err
			}
			ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
			if err != nil {
				return nil, errBlockedRemoteHost
			}
			var public []net.IP
			for _, ipa := range ips {
				if !isBlockedRemoteIP(ipa.IP) {
					public = append(public, ipa.IP)
				}
			}
			if len(public) == 0 {
				return nil, errBlockedRemoteHost
			}
			d := &net.Dialer{Timeout: 10 * time.Second}
			return d.DialContext(ctx, network, net.JoinHostPort(public[0].String(), port))
		},
	},
}

// downloadRemoteImage fetches a remote http(s) image for
// input_reference.image_url, validating scheme, destination, status, content
// type, and size. Client-facing errors are deliberately generic: echoing
// upstream statuses/content-types would give an internal network
// fingerprinting oracle.
func downloadRemoteImage(ctx context.Context, u string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, errors.New("input_reference.image_url: invalid URL")
	}
	resp, err := remoteImageClient.Do(req)
	if err != nil {
		if errors.Is(err, errBlockedRemoteHost) {
			return nil, errBlockedRemoteHost
		}
		return nil, errors.New("input_reference.image_url: download failed")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("input_reference.image_url: download failed")
	}
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		mediaType, _, _ := strings.Cut(ct, ";")
		if !strings.HasPrefix(strings.TrimSpace(mediaType), "image/") {
			return nil, errors.New("input_reference.image_url: URL did not return an image")
		}
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxRemoteImageBytes+1))
	if err != nil {
		return nil, errors.New("input_reference.image_url: download failed")
	}
	if int64(len(data)) > maxRemoteImageBytes {
		return nil, fmt.Errorf("input_reference.image_url: image exceeds the %d byte limit", maxRemoteImageBytes)
	}
	if len(data) == 0 {
		return nil, errors.New("input_reference.image_url: download was empty")
	}
	return data, nil
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

// fromForm parses the `video` reference from a multipart form: a file part
// (uploaded MP4), a JSON object form field, or the SDK's bracket-notation
// field video[id].
func (v *videoSourceInput) fromForm(r *http.Request) error {
	var ref openai.VideoReferenceParam
	data, err := parseMultipartRefField(r, "video", &ref, map[string]*string{
		"id": &v.videoID,
	})
	if err != nil {
		return err
	}
	v.videoFile = data
	if ref.ID != "" {
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
// resolves the `video` reference, and when that reference is an {id} inherits
// model/size from the source job (the SDK omits them, like the cloud API) and
// captures its Seconds (for stitched-total reporting on extensions).
// secondsRequired controls whether `seconds` is mandatory (extensions) or
// optional with a default (edits); secondsValues selects the allowed set
// (VideoSecondsValues for edits, VideoExtensionSecondsValues for extensions).
// On any validation failure it aborts the gin context with a 400/503 and
// returns ok=false.
func (s *Server) parseEditExtendRequest(c *gin.Context, secondsRequired bool, secondsValues map[string]bool) (editExtendInput, bool) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return editExtendInput{}, false
	}

	const maxVideoBodyBytes int64 = 256 << 20 // 256 MiB (accommodate uploaded source MP4)
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxVideoBodyBytes)

	in := editExtendInput{}

	contentType := c.GetHeader("Content-Type")
	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := c.Request.ParseMultipartForm(32 << 20); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to parse multipart form: "+err.Error()))
			return editExtendInput{}, false
		}
		in.prompt = c.Request.FormValue("prompt")
		in.model = c.Request.FormValue("model")
		in.seconds = c.Request.FormValue("seconds")
		in.size = c.Request.FormValue("size")
		if err := in.src.fromForm(c.Request); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return editExtendInput{}, false
		}
	} else {
		// JSON fallback (aiollama extension; the spec content type is
		// multipart). The video source must be an {"id": ...} reference —
		// file upload is only possible via multipart.
		var body struct {
			Prompt  string                     `json:"prompt"`
			Model   string                     `json:"model,omitempty"`
			Seconds string                     `json:"seconds,omitempty"`
			Size    string                     `json:"size,omitempty"`
			Video   openai.VideoReferenceParam `json:"video,omitempty"`
		}
		if err := c.ShouldBindJSON(&body); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return editExtendInput{}, false
		}
		in.prompt = body.Prompt
		in.model = body.Model
		in.seconds = body.Seconds
		in.size = body.Size
		in.src.videoID = body.Video.ID
	}

	if !validateVideoPrompt(c, in.prompt) {
		return editExtendInput{}, false
	}

	if !in.src.hasVideo() {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "video is required (a file part or a {id} object)"))
		return editExtendInput{}, false
	}

	// The openai-python SDK sends neither model nor size on edits/extensions
	// (the cloud API inherits them from the source video). When the source is
	// an {id} reference, resolve the source job once — before model/size
	// validation — and inherit the fields the client omitted. This lookup
	// also captures the source's Seconds (for stitched-total reporting on
	// extensions).
	if in.src.videoID != "" {
		if job, err := s.videoJobs.Get(in.src.videoID); err == nil {
			src := job.ToVideo()
			if in.model == "" {
				in.model = src.Model
			}
			if in.size == "" {
				in.size = src.Size
			}
			in.sourceSeconds = src.Seconds
		}
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

	// For an uploaded file on an extension, probe the duration so the
	// stitched-total seconds is reported correctly. (For an {id} reference
	// the source job's Seconds was already captured above.)
	if in.src.videoID == "" && secondsRequired && len(in.src.videoFile) > 0 {
		if tc := s.videoJobs.Transcoder(); tc != nil && tc.Available() {
			if secs, err := tc.ProbeDurationSeconds(c.Request.Context(), in.src.videoFile); err == nil && secs > 0 {
				in.sourceSeconds = strconv.Itoa(secs)
			} else {
				slog.Warn("could not probe uploaded source video duration; stitched seconds will report requested seconds only", "error", err)
			}
		}
	}

	in.width, in.height = videoSizeDimensions(in.size)

	return in, true
}

// createSourceBasedJob creates the async job shared by the edit, extend, and
// remix handlers: a re-render driven from a source video (a referenced job
// and/or uploaded bytes). extend selects the extension semantics (last-frame
// init + source concatenation) versus the edit/remix semantics (first-frame
// init, new standalone clip).
func (s *Server) createSourceBasedJob(c *gin.Context, in editExtendInput, extend bool) {
	job, err := s.videoJobs.Create(videojobs.CreateParams{
		Model:         in.model,
		Prompt:        in.prompt,
		Seconds:       in.seconds,
		Size:          in.size,
		RemixedFromID: in.src.videoID,
		SourceVideoID: in.src.videoID,
		SourceVideo:   in.src.videoFile,
		Extend:        extend,
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
	s.createSourceBasedJob(c, in, false)
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
	s.createSourceBasedJob(c, in, true)
}

// VideoRemixHandler handles POST /v1/videos/{video_id}/remix. The request
// body is JSON {"prompt": "..."} (the SDK's VideoRemixParams carries no file
// params, hence no multipart). Remix is semantically an edit by id with a
// new prompt: it re-renders a new video from the source's first frame as an
// I2V init image, inheriting model/size/seconds from the source job. The
// response Video.remixed_from_video_id references the source.
//
// Spec: https://developers.openai.com/api/reference/resources/videos/remix
func (s *Server) VideoRemixHandler(c *gin.Context) {
	if s.videoJobs == nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, openai.NewError(http.StatusServiceUnavailable, "video generation is not available"))
		return
	}

	id := c.Param("video_id")
	src, err := s.videoJobs.Get(id)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "video not found: "+id))
		return
	}
	srcVideo := src.ToVideo()
	if srcVideo.Status != openai.VideoStatusCompleted {
		c.AbortWithStatusJSON(http.StatusConflict, openai.NewError(http.StatusConflict, fmt.Sprintf("video is not completed (status: %s)", srcVideo.Status)))
		return
	}

	const maxRemixBodyBytes int64 = 1 << 20 // 1 MiB (prompt-only JSON)
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxRemixBodyBytes)
	var body struct {
		Prompt string `json:"prompt"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}
	if !validateVideoPrompt(c, body.Prompt) {
		return
	}

	w, h := videoSizeDimensions(srcVideo.Size)
	s.createSourceBasedJob(c, editExtendInput{
		prompt:        body.Prompt,
		model:         srcVideo.Model,
		seconds:       srcVideo.Seconds,
		size:          srcVideo.Size,
		src:           videoSourceInput{videoID: id},
		sourceSeconds: srcVideo.Seconds,
		width:         w,
		height:        h,
	}, false)
}

// VideoCharactersHandler responds 501 Not Implemented for the Sora cloud
// characters endpoints (POST /v1/videos/characters, GET
// /v1/videos/characters/{id}). Persistent characters are a cloud feature
// with no local SD.cpp/WAN/LTX equivalent; the local character-consistency
// use case is covered by LoRAs and input_reference.
func (s *Server) VideoCharactersHandler(c *gin.Context) {
	c.AbortWithStatusJSON(http.StatusNotImplemented, openai.NewError(http.StatusNotImplemented, "characters are a Sora cloud feature; not supported by aiollama"))
}
