package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

// This file implements the OpenAI Images API surface:
//   - POST /v1/images/generations (application/json)
//   - POST /v1/images/edits (multipart/form-data per spec; JSON as an
//     aiollama extension for backwards compatibility)
//   - GET  /v1/images/files/{id} (download for response_format=url)
//
// Spec: https://developers.openai.com/api/reference/resources/images
//
// Unlike the chat/completions endpoints, these are NOT implemented as a
// middleware rewriting into /api/generate: the spec requires features the
// single-shot generate pipeline cannot express (n>1 images per request,
// per-request output transcoding, usage block, URL delivery), so the
// handlers drive the scheduler/runner directly, following the same pattern
// as buildVideoGenerateFunc in videoapi.go.

// imageGenParams is the normalized, validated image request shared by the
// generations and edits handlers.
type imageGenParams struct {
	model  string
	prompt string
	// images are the decoded input/reference images (edits only).
	images [][]byte
	// mask is the inpainting mask converted to SD.cpp semantics (opaque
	// gray PNG, white = region to edit). Nil when absent.
	mask              []byte
	n                 int
	width, height     int32
	steps             int32
	responseFormat    string
	outputFormat      string
	outputCompression *int
	seed              *int64
}

// maxImageBodyBytes bounds image request bodies (16 inputs x ~25 MiB).
const maxImageBodyBytes int64 = 512 << 20

// ImageGenerationsHandler handles POST /v1/images/generations. The spec
// content type is application/json.
func (s *Server) ImageGenerationsHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxImageBodyBytes)

	var req openai.ImageGenerationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
		return
	}

	params, ok := validateImageRequest(c, req.Model, req.Prompt, req.N, req.Size, req.Quality, req.ResponseFormat, req.OutputFormat, req.OutputCompression, req.Background, req.Style, req.Moderation, req.Stream, req.Seed)
	if !ok {
		return
	}

	s.runImageGeneration(c, params)
}

// ImageEditsHandler handles POST /v1/images/edits. The spec content type is
// multipart/form-data with an `image` file part (or repeated `image[]`
// parts for multiple inputs) and an optional `mask` file part. A JSON body
// with base64/data-URL images is also accepted (aiollama extension).
func (s *Server) ImageEditsHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxImageBodyBytes)

	var req openai.ImageEditRequest

	contentType := c.GetHeader("Content-Type")
	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := c.Request.ParseMultipartForm(64 << 20); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to parse multipart form: "+err.Error()))
			return
		}
		if err := imageEditFromMultipart(c.Request, &req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
	} else {
		if err := imageEditFromJSON(c.Request, &req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}
	}

	params, ok := validateImageRequest(c, req.Model, req.Prompt, req.N, req.Size, req.Quality, req.ResponseFormat, req.OutputFormat, req.OutputCompression, req.Background, "", "", false, req.Seed)
	if !ok {
		return
	}
	if len(req.Images) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "image is required"))
		return
	}
	params.images = make([][]byte, 0, len(req.Images))
	for _, img := range req.Images {
		params.images = append(params.images, img)
	}
	params.mask = req.Mask

	s.runImageGeneration(c, params)
}

// ImageFileHandler handles GET /v1/images/files/{image_id}. It serves
// images previously generated with response_format=url until their TTL
// expires.
func (s *Server) ImageFileHandler(c *gin.Context) {
	if s.imageFiles == nil {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "image not found"))
		return
	}
	id := c.Param("image_id")
	data, contentType, ok := s.imageFiles.get(id)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, "image not found: "+id))
		return
	}
	ext := "png"
	switch contentType {
	case "image/jpeg":
		ext = "jpg"
	case "image/webp":
		ext = "webp"
	}
	c.Header("Content-Disposition", fmt.Sprintf("inline; filename=\"%s.%s\"", id, ext))
	c.Data(http.StatusOK, contentType, data)
}

// validateImageRequest applies the spec's shared scalar validation and
// defaults for generations and edits. On failure it aborts with a 400 and
// returns ok=false. style/moderation/user are validated for enum
// conformance but otherwise have no local effect.
func validateImageRequest(c *gin.Context, modelName, prompt string, n int, size, quality, responseFormat, outputFormat string, outputCompression *int, background, style, moderation string, stream bool, seed *int64) (imageGenParams, bool) {
	bad := func(msg string) (imageGenParams, bool) {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, msg))
		return imageGenParams{}, false
	}

	if prompt == "" {
		return bad("prompt is required")
	}
	if modelName == "" {
		return bad("model is required")
	}
	if stream {
		return bad("streaming (stream=true with partial_images) is not supported by this server")
	}

	if n == 0 {
		n = 1
	}
	if n < 1 || n > openai.ImageMaxN {
		return bad(fmt.Sprintf("n must be between 1 and %d", openai.ImageMaxN))
	}

	if size == "" {
		size = openai.ImageDefaultSize
	}
	w, h, err := openai.ParseImageSize(size)
	if err != nil {
		return bad(err.Error())
	}

	if quality == "" {
		quality = openai.ImageQualityAuto
	} else if !openai.ValidImageQuality(quality) {
		return bad(fmt.Sprintf("quality must be one of low, medium, high, auto; got %q", quality))
	}

	if responseFormat == "" {
		responseFormat = openai.ImageResponseFormatB64JSON
	} else if responseFormat != openai.ImageResponseFormatB64JSON && responseFormat != openai.ImageResponseFormatURL {
		return bad(fmt.Sprintf("response_format must be one of b64_json, url; got %q", responseFormat))
	}

	if outputFormat == "" {
		outputFormat = openai.ImageOutputFormatPNG
	} else if outputFormat != openai.ImageOutputFormatPNG && outputFormat != openai.ImageOutputFormatJPEG && outputFormat != openai.ImageOutputFormatWebP {
		return bad(fmt.Sprintf("output_format must be one of png, jpeg, webp; got %q", outputFormat))
	}
	if outputFormat == openai.ImageOutputFormatWebP && !webpEncodeAvailable() {
		return bad("output_format \"webp\" requires ffmpeg on the server's PATH")
	}

	if outputCompression != nil && (*outputCompression < 0 || *outputCompression > 100) {
		return bad("output_compression must be between 0 and 100")
	}

	if background != "" && background != openai.ImageBackgroundTransparent && background != openai.ImageBackgroundOpaque && background != openai.ImageBackgroundAuto {
		return bad(fmt.Sprintf("background must be one of transparent, opaque, auto; got %q", background))
	}
	if style != "" && style != "vivid" && style != "natural" {
		return bad(fmt.Sprintf("style must be one of vivid, natural; got %q", style))
	}
	if moderation != "" && moderation != "low" && moderation != "auto" {
		return bad(fmt.Sprintf("moderation must be one of low, auto; got %q", moderation))
	}

	return imageGenParams{
		model:             modelName,
		prompt:            prompt,
		n:                 n,
		width:             w,
		height:            h,
		steps:             openai.StepsForImageQuality(quality),
		responseFormat:    responseFormat,
		outputFormat:      outputFormat,
		outputCompression: outputCompression,
		seed:              seed,
	}, true
}

// runImageGeneration executes n image generations through the scheduler and
// writes the OpenAI Images API response.
func (s *Server) runImageGeneration(c *gin.Context, p imageGenParams) {
	ctx := c.Request.Context()

	modelLoaded, err := GetModel(p.model)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, openai.NewError(http.StatusNotFound, fmt.Sprintf("model '%s' not found", p.model)))
		return
	}
	if err := modelLoaded.CheckCapabilities(model.CapabilityImage); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("%q does not support image generation", p.model)))
		return
	}

	runner, m, _, err := s.scheduleRunner(ctx, p.model, []model.Capability{model.CapabilityImage}, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, p.model, err)
		return
	}

	var media []llm.MediaData
	for i, imgData := range p.images {
		media = append(media, llm.NewMediaData(i, imgData))
	}

	data := make([]openai.ImageURLOrData, 0, p.n)
	inputTokens := 0
	for i := 0; i < p.n; i++ {
		var seed int64
		if p.seed != nil {
			// Distinct deterministic seed per image when one was requested.
			seed = *p.seed + int64(i)
		}

		var imgB64 string
		err := runner.Completion(ctx, llm.CompletionRequest{
			Prompt: p.prompt,
			Width:  p.width,
			Height: p.height,
			Steps:  p.steps,
			Seed:   seed,
			Media:  media,
			Mask:   p.mask,
		}, func(cr llm.CompletionResponse) {
			if cr.Image != "" {
				imgB64 = cr.Image
			}
			if cr.Done {
				inputTokens += cr.PromptEvalCount
			}
		})
		if err != nil {
			s.sched.expireRunnersForRuntimeOOM(m, err)
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}
		if imgB64 == "" {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, "image generation produced no image"))
			return
		}

		pngBytes, err := base64.StdEncoding.DecodeString(imgB64)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, "runner returned invalid image data"))
			return
		}

		out, contentType, err := transcodeOutputImage(ctx, pngBytes, p.outputFormat, p.outputCompression)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		if p.responseFormat == openai.ImageResponseFormatURL {
			id := s.imageFiles.put(out, contentType)
			data = append(data, openai.ImageURLOrData{URL: requestBaseURL(c) + "/v1/images/files/" + id})
		} else {
			data = append(data, openai.ImageURLOrData{B64JSON: base64.StdEncoding.EncodeToString(out)})
		}
	}

	c.JSON(http.StatusOK, openai.ImageGenerationResponse{
		Created: time.Now().Unix(),
		Data:    data,
		Usage: &openai.ImageUsage{
			InputTokens:  inputTokens,
			OutputTokens: 0,
			TotalTokens:  inputTokens,
		},
	})
}

// requestBaseURL builds the absolute base URL for this request, honoring
// reverse-proxy headers, for response_format=url download links.
func requestBaseURL(c *gin.Context) string {
	scheme := "http"
	if c.Request.TLS != nil {
		scheme = "https"
	}
	if proto := c.GetHeader("X-Forwarded-Proto"); proto != "" {
		scheme = proto
	}
	return scheme + "://" + c.Request.Host
}

// transcodeOutputImage converts the runner's PNG output to the requested
// OpenAI output_format, returning the encoded bytes and content type.
// PNG is passed through untouched; JPEG uses the stdlib encoder (quality =
// output_compression, default 100 per spec); WebP requires ffmpeg.
func transcodeOutputImage(ctx context.Context, pngBytes []byte, format string, quality *int) ([]byte, string, error) {
	switch format {
	case "", openai.ImageOutputFormatPNG:
		return pngBytes, "image/png", nil
	case openai.ImageOutputFormatJPEG:
		img, err := png.Decode(bytes.NewReader(pngBytes))
		if err != nil {
			return nil, "", fmt.Errorf("decode runner image: %w", err)
		}
		q := 100
		if quality != nil {
			q = *quality
			if q < 1 {
				q = 1
			}
		}
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: q}); err != nil {
			return nil, "", fmt.Errorf("encode jpeg: %w", err)
		}
		return buf.Bytes(), "image/jpeg", nil
	case openai.ImageOutputFormatWebP:
		out, err := encodeWebP(ctx, pngBytes, quality)
		if err != nil {
			return nil, "", err
		}
		return out, "image/webp", nil
	}
	return nil, "", fmt.Errorf("unsupported output_format %q", format)
}

// imageFFmpeg caches the ffmpeg lookup used for WebP encoding.
var imageFFmpeg struct {
	once sync.Once
	path string
	err  error
}

func lookupImageFFmpeg() (string, error) {
	imageFFmpeg.once.Do(func() {
		imageFFmpeg.path, imageFFmpeg.err = exec.LookPath("ffmpeg")
	})
	return imageFFmpeg.path, imageFFmpeg.err
}

// webpEncodeAvailable reports whether WebP output transcoding is possible.
func webpEncodeAvailable() bool {
	_, err := lookupImageFFmpeg()
	return err == nil
}

// encodeWebP transcodes a PNG to lossy WebP via ffmpeg. quality maps to
// ffmpeg's -quality (0-100, default 100 per the OpenAI spec).
func encodeWebP(ctx context.Context, pngBytes []byte, quality *int) ([]byte, error) {
	ffmpeg, err := lookupImageFFmpeg()
	if err != nil {
		return nil, fmt.Errorf("output_format \"webp\" requires ffmpeg on PATH: %w", err)
	}
	q := 100
	if quality != nil {
		q = *quality
	}

	args := []string{
		"-hide_banner", "-loglevel", "error",
		"-f", "image2pipe", "-vcodec", "png", "-i", "pipe:0",
		"-quality", strconv.Itoa(q),
		"-frames:v", "1",
		"-f", "webp", "pipe:1",
	}
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	cmd.Stdin = bytes.NewReader(pngBytes)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("webp encoding failed: %s", msg)
	}
	if stdout.Len() == 0 {
		return nil, errors.New("webp encoding produced no output")
	}
	return stdout.Bytes(), nil
}

// imageEditFromMultipart parses the spec multipart form of POST
// /v1/images/edits into req. Images arrive as file parts named "image"
// (single) or "image[]" (multiple), or as base64/data-URL form values.
// The mask (optional) is converted to SD.cpp semantics.
func imageEditFromMultipart(r *http.Request, req *openai.ImageEditRequest) error {
	req.Model = r.FormValue("model")
	req.Prompt = r.FormValue("prompt")
	req.Size = r.FormValue("size")
	req.Quality = r.FormValue("quality")
	req.ResponseFormat = r.FormValue("response_format")
	req.Background = r.FormValue("background")
	req.OutputFormat = r.FormValue("output_format")
	req.User = r.FormValue("user")

	if v := r.FormValue("n"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid n: %q", v)
		}
		req.N = n
	}
	if v := r.FormValue("output_compression"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid output_compression: %q", v)
		}
		req.OutputCompression = &n
	}
	if v := r.FormValue("seed"); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("invalid seed: %q", v)
		}
		req.Seed = &n
	}

	// File parts: "image" (dall-e-2 style single) and "image[]" (gpt-image-1
	// style array). Both are accepted regardless of count.
	if r.MultipartForm != nil {
		for _, key := range []string{"image", "image[]"} {
			for _, fh := range r.MultipartForm.File[key] {
				f, err := fh.Open()
				if err != nil {
					return fmt.Errorf("open image file: %w", err)
				}
				data, err := io.ReadAll(f)
				f.Close()
				if err != nil {
					return fmt.Errorf("read image file: %w", err)
				}
				if len(data) > 0 {
					req.Images = append(req.Images, api.ImageData(data))
				}
			}
		}
	}
	// Base64/data-URL form values ("image" may repeat for arrays).
	if r.MultipartForm != nil {
		for _, key := range []string{"image", "image[]"} {
			for _, v := range r.MultipartForm.Value[key] {
				if v == "" {
					continue
				}
				img, err := openai.DecodeImageDataURL(v)
				if err != nil {
					return fmt.Errorf("invalid image: %w", err)
				}
				req.Images = append(req.Images, img)
			}
		}
	}
	if len(req.Images) > openai.ImageMaxEditInputs {
		return fmt.Errorf("too many images: got %d, maximum is %d", len(req.Images), openai.ImageMaxEditInputs)
	}

	// Optional mask: file part or base64 form value.
	var maskRaw []byte
	if r.MultipartForm != nil {
		if fhs := r.MultipartForm.File["mask"]; len(fhs) > 0 {
			f, err := fhs[0].Open()
			if err != nil {
				return fmt.Errorf("open mask file: %w", err)
			}
			defer f.Close()
			maskRaw, err = io.ReadAll(f)
			if err != nil {
				return fmt.Errorf("read mask file: %w", err)
			}
		} else if v := r.FormValue("mask"); v != "" {
			img, err := openai.DecodeImageDataURL(v)
			if err != nil {
				return fmt.Errorf("invalid mask: %w", err)
			}
			maskRaw = img
		}
	}
	if len(maskRaw) > 0 {
		mask, err := ConvertMaskToSDCPP(maskRaw)
		if err != nil {
			return fmt.Errorf("invalid mask: %w", err)
		}
		req.Mask = mask
	}
	return nil
}

// imageEditFromJSON parses the JSON extension form of POST /v1/images/edits.
// `image` may be a single base64/data-URL string or an array of them; `mask`
// is a single base64/data-URL string.
func imageEditFromJSON(r *http.Request, req *openai.ImageEditRequest) error {
	var body struct {
		Model             string          `json:"model"`
		Prompt            string          `json:"prompt"`
		Image             json.RawMessage `json:"image"`
		Mask              string          `json:"mask"`
		N                 int             `json:"n,omitempty"`
		Size              string          `json:"size,omitempty"`
		Quality           string          `json:"quality,omitempty"`
		ResponseFormat    string          `json:"response_format,omitempty"`
		Background        string          `json:"background,omitempty"`
		OutputFormat      string          `json:"output_format,omitempty"`
		OutputCompression *int            `json:"output_compression,omitempty"`
		User              string          `json:"user,omitempty"`
		Seed              *int64          `json:"seed,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		return err
	}

	req.Model = body.Model
	req.Prompt = body.Prompt
	req.N = body.N
	req.Size = body.Size
	req.Quality = body.Quality
	req.ResponseFormat = body.ResponseFormat
	req.Background = body.Background
	req.OutputFormat = body.OutputFormat
	req.OutputCompression = body.OutputCompression
	req.User = body.User
	req.Seed = body.Seed

	if len(body.Image) > 0 && string(body.Image) != "null" {
		var images []string
		if err := json.Unmarshal(body.Image, &images); err != nil {
			var single string
			if err := json.Unmarshal(body.Image, &single); err != nil {
				return errors.New("image must be a base64 string or an array of base64 strings")
			}
			images = []string{single}
		}
		for _, v := range images {
			img, err := openai.DecodeImageDataURL(v)
			if err != nil {
				return fmt.Errorf("invalid image: %w", err)
			}
			req.Images = append(req.Images, img)
		}
		if len(req.Images) > openai.ImageMaxEditInputs {
			return fmt.Errorf("too many images: got %d, maximum is %d", len(req.Images), openai.ImageMaxEditInputs)
		}
	}

	if body.Mask != "" {
		maskRaw, err := openai.DecodeImageDataURL(body.Mask)
		if err != nil {
			return fmt.Errorf("invalid mask: %w", err)
		}
		mask, err := ConvertMaskToSDCPP(maskRaw)
		if err != nil {
			return fmt.Errorf("invalid mask: %w", err)
		}
		req.Mask = mask
	}
	return nil
}

// ConvertMaskToSDCPP converts an OpenAI edit mask to SD.cpp inpainting mask
// semantics. OpenAI: fully transparent pixels (alpha = 0) mark the region to
// edit. SD.cpp: white (255) marks the region to edit. If the mask has no
// meaningful alpha channel, it is treated as an SD-native mask already
// (bright pixels = edit region). The result is an opaque grayscale PNG.
func ConvertMaskToSDCPP(maskBytes []byte) ([]byte, error) {
	img, _, err := image.Decode(bytes.NewReader(maskBytes))
	if err != nil {
		return nil, fmt.Errorf("decode mask: %w", err)
	}
	b := img.Bounds()

	// Detect whether the mask carries meaningful transparency.
	hasAlpha := false
scan:
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if _, _, _, a := img.At(x, y).RGBA(); a != 0xffff {
				hasAlpha = true
				break scan
			}
		}
	}

	gray := image.NewGray(b)
	white := color.Gray{Y: 255}
	black := color.Gray{Y: 0}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if hasAlpha {
				_, _, _, a := img.At(x, y).RGBA()
				if a < 0x8000 {
					gray.SetGray(x, y, white)
				} else {
					gray.SetGray(x, y, black)
				}
			} else {
				// SD-native semantics: bright = edit.
				r, g, bl, _ := img.At(x, y).RGBA()
				lum := (299*r + 587*g + 114*bl) / 1000
				if lum >= 0x8000 {
					gray.SetGray(x, y, white)
				} else {
					gray.SetGray(x, y, black)
				}
			}
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, gray); err != nil {
		return nil, fmt.Errorf("encode mask: %w", err)
	}
	return buf.Bytes(), nil
}
