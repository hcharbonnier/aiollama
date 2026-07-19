package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/openai"
)

// setupImageRouter binds the image handlers on a Server with no scheduler:
// validation errors (400) and model-not-found (404) are reachable without
// any models on disk.
func setupImageRouter(t *testing.T) (*Server, *gin.Engine) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	s := &Server{imageFiles: newImageFileStore()}
	r := gin.New()
	r.POST("/v1/images/generations", s.ImageGenerationsHandler)
	r.POST("/v1/images/edits", s.ImageEditsHandler)
	r.GET("/v1/images/files/:image_id", s.ImageFileHandler)
	return s, r
}

// tinyPNG returns a 2x2 opaque PNG.
func tinyPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := range img.Pix {
		img.Pix[i] = 0x7f
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestImageGenerationsValidation(t *testing.T) {
	_, r := setupImageRouter(t)

	tests := []struct {
		name     string
		body     string
		wantCode int
		wantMsg  string
	}{
		{name: "missing prompt", body: `{"model": "m"}`, wantCode: 400, wantMsg: "prompt is required"},
		{name: "missing model", body: `{"prompt": "p"}`, wantCode: 400, wantMsg: "model is required"},
		{name: "empty body", body: `{}`, wantCode: 400, wantMsg: "prompt is required"},
		{name: "n too large", body: `{"model": "m", "prompt": "p", "n": 11}`, wantCode: 400, wantMsg: "n must be between 1 and 10"},
		{name: "n negative", body: `{"model": "m", "prompt": "p", "n": -1}`, wantCode: 400, wantMsg: "n must be between 1 and 10"},
		{name: "bad size", body: `{"model": "m", "prompt": "p", "size": "big"}`, wantCode: 400, wantMsg: "invalid size"},
		{name: "oversized", body: `{"model": "m", "prompt": "p", "size": "8192x8192"}`, wantCode: 400, wantMsg: "maximum dimension"},
		{name: "bad quality", body: `{"model": "m", "prompt": "p", "quality": "ultra"}`, wantCode: 400, wantMsg: "quality must be one of"},
		{name: "quality alias standard", body: `{"model": "m", "prompt": "p", "quality": "standard"}`, wantCode: 404},
		{name: "quality alias hd", body: `{"model": "m", "prompt": "p", "quality": "hd"}`, wantCode: 404},
		{name: "partial_images unsupported", body: `{"model": "m", "prompt": "p", "partial_images": 2}`, wantCode: 400, wantMsg: "streaming"},
		{name: "bad response_format", body: `{"model": "m", "prompt": "p", "response_format": "xml"}`, wantCode: 400, wantMsg: "response_format must be one of"},
		{name: "bad output_format", body: `{"model": "m", "prompt": "p", "output_format": "tiff"}`, wantCode: 400, wantMsg: "output_format must be one of"},
		{name: "bad output_compression", body: `{"model": "m", "prompt": "p", "output_compression": 101}`, wantCode: 400, wantMsg: "output_compression must be between 0 and 100"},
		{name: "bad background", body: `{"model": "m", "prompt": "p", "background": "green"}`, wantCode: 400, wantMsg: "background must be one of"},
		{name: "bad style", body: `{"model": "m", "prompt": "p", "style": "anime"}`, wantCode: 400, wantMsg: "style must be one of"},
		{name: "bad moderation", body: `{"model": "m", "prompt": "p", "moderation": "strict"}`, wantCode: 400, wantMsg: "moderation must be one of"},
		{name: "stream unsupported", body: `{"model": "m", "prompt": "p", "stream": true}`, wantCode: 400, wantMsg: "streaming"},
		// Valid requests pass validation and fail at model lookup (404),
		// proving the full scalar surface is accepted.
		{name: "valid minimal", body: `{"model": "m", "prompt": "p"}`, wantCode: 404},
		{name: "valid full", body: `{"model": "m", "prompt": "p", "n": 2, "size": "512x768", "quality": "high", "response_format": "url", "output_format": "jpeg", "output_compression": 80, "background": "opaque", "style": "vivid", "seed": 42, "user": "u"}`, wantCode: 404},
		{name: "valid free size", body: `{"model": "m", "prompt": "p", "size": "640x360"}`, wantCode: 404},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)

			if w.Code != tt.wantCode {
				t.Fatalf("status = %d, want %d: %s", w.Code, tt.wantCode, w.Body.String())
			}
			var errResp openai.ErrorResponse
			if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
				t.Fatalf("error body is not OpenAI format: %v", err)
			}
			if tt.wantMsg != "" && !strings.Contains(w.Body.String(), tt.wantMsg) {
				t.Errorf("body %q does not contain %q", w.Body.String(), tt.wantMsg)
			}
		})
	}
}

// newImageEditMultipart builds a multipart POST /v1/images/edits request.
func newImageEditMultipart(t *testing.T, fields map[string]string, imageParts []string, withMask bool) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	for _, field := range imageParts {
		fw, err := mw.CreateFormFile(field, "img.png")
		if err != nil {
			t.Fatal(err)
		}
		fw.Write(tinyPNG(t))
	}
	if withMask {
		fw, err := mw.CreateFormFile("mask", "mask.png")
		if err != nil {
			t.Fatal(err)
		}
		fw.Write(tinyPNG(t))
	}
	_ = mw.Close()
	req, err := http.NewRequest(http.MethodPost, "/v1/images/edits", &body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

func TestImageEditsParsing(t *testing.T) {
	_, r := setupImageRouter(t)

	tests := []struct {
		name     string
		build    func() *http.Request
		wantCode int
		wantMsg  string
	}{
		{
			name: "multipart missing image",
			build: func() *http.Request {
				return newImageEditMultipart(t, map[string]string{"model": "m", "prompt": "p"}, nil, false)
			},
			wantCode: 400, wantMsg: "image is required",
		},
		{
			name: "multipart single image file",
			build: func() *http.Request {
				return newImageEditMultipart(t, map[string]string{"model": "m", "prompt": "p"}, []string{"image"}, false)
			},
			wantCode: 404, // parsing succeeded; model lookup fails
		},
		{
			name: "multipart image[] array",
			build: func() *http.Request {
				return newImageEditMultipart(t, map[string]string{"model": "m", "prompt": "p"}, []string{"image[]", "image[]"}, false)
			},
			wantCode: 404,
		},
		{
			name: "multipart with mask",
			build: func() *http.Request {
				return newImageEditMultipart(t, map[string]string{"model": "m", "prompt": "p"}, []string{"image"}, true)
			},
			wantCode: 404,
		},
		{
			name: "multipart missing prompt",
			build: func() *http.Request {
				return newImageEditMultipart(t, map[string]string{"model": "m"}, []string{"image"}, false)
			},
			wantCode: 400, wantMsg: "prompt is required",
		},
		{
			name: "multipart bad n",
			build: func() *http.Request {
				return newImageEditMultipart(t, map[string]string{"model": "m", "prompt": "p", "n": "abc"}, []string{"image"}, false)
			},
			wantCode: 400, wantMsg: "invalid n",
		},
		{
			name: "json single image",
			build: func() *http.Request {
				b64 := base64.StdEncoding.EncodeToString(tinyPNG(t))
				body := `{"model": "m", "prompt": "p", "image": "data:image/png;base64,` + b64 + `"}`
				req, _ := http.NewRequest(http.MethodPost, "/v1/images/edits", strings.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				return req
			},
			wantCode: 404,
		},
		{
			name: "json image array",
			build: func() *http.Request {
				b64 := "data:image/png;base64," + base64.StdEncoding.EncodeToString(tinyPNG(t))
				body := `{"model": "m", "prompt": "p", "image": ["` + b64 + `", "` + b64 + `"]}`
				req, _ := http.NewRequest(http.MethodPost, "/v1/images/edits", strings.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				return req
			},
			wantCode: 404,
		},
		{
			name: "json invalid image",
			build: func() *http.Request {
				body := `{"model": "m", "prompt": "p", "image": "not-base64!!!"}`
				req, _ := http.NewRequest(http.MethodPost, "/v1/images/edits", strings.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				return req
			},
			wantCode: 400, wantMsg: "invalid image",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			r.ServeHTTP(w, tt.build())
			if w.Code != tt.wantCode {
				t.Fatalf("status = %d, want %d: %s", w.Code, tt.wantCode, w.Body.String())
			}
			if tt.wantMsg != "" && !strings.Contains(w.Body.String(), tt.wantMsg) {
				t.Errorf("body %q does not contain %q", w.Body.String(), tt.wantMsg)
			}
		})
	}
}

func TestConvertMaskToSDCPP(t *testing.T) {
	// Mask with a transparent left half (edit region) and opaque right half
	// (keep region). OpenAI semantics: alpha=0 → edit.
	img := image.NewNRGBA(image.Rect(0, 0, 4, 1))
	for x := 0; x < 2; x++ {
		img.SetNRGBA(x, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 0})
	}
	for x := 2; x < 4; x++ {
		img.SetNRGBA(x, 0, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}

	out, err := ConvertMaskToSDCPP(buf.Bytes())
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	gray, err := png.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode converted mask: %v", err)
	}
	// SD.cpp semantics: white = edit. Left half (was transparent) → white.
	for x := 0; x < 2; x++ {
		r, g, b, _ := gray.At(x, 0).RGBA()
		if r>>8 != 255 || g>>8 != 255 || b>>8 != 255 {
			t.Errorf("pixel %d: expected white (edit), got %d,%d,%d", x, r>>8, g>>8, b>>8)
		}
	}
	for x := 2; x < 4; x++ {
		r, g, b, _ := gray.At(x, 0).RGBA()
		if r>>8 != 0 || g>>8 != 0 || b>>8 != 0 {
			t.Errorf("pixel %d: expected black (keep), got %d,%d,%d", x, r>>8, g>>8, b>>8)
		}
	}

	// Opaque SD-native mask: white = edit, preserved through conversion.
	native := image.NewGray(image.Rect(0, 0, 2, 1))
	native.SetGray(0, 0, color.Gray{Y: 255})
	native.SetGray(1, 0, color.Gray{Y: 0})
	buf.Reset()
	if err := png.Encode(&buf, native); err != nil {
		t.Fatal(err)
	}
	out, err = ConvertMaskToSDCPP(buf.Bytes())
	if err != nil {
		t.Fatalf("convert native: %v", err)
	}
	gray, err = png.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	if r, _, _, _ := gray.At(0, 0).RGBA(); r>>8 != 255 {
		t.Errorf("native white pixel lost: got %d", r>>8)
	}
	if r, _, _, _ := gray.At(1, 0).RGBA(); r>>8 != 0 {
		t.Errorf("native black pixel lost: got %d", r>>8)
	}
}

func TestImageFileStoreRoundTrip(t *testing.T) {
	s, r := setupImageRouter(t)

	id := s.imageFiles.put([]byte("png-bytes"), "image/png")
	if !strings.HasPrefix(id, "img_") {
		t.Errorf("id %q missing img_ prefix", id)
	}

	req, _ := http.NewRequest(http.MethodGet, "/v1/images/files/"+id, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if w.Header().Get("Content-Type") != "image/png" {
		t.Errorf("content-type = %q, want image/png", w.Header().Get("Content-Type"))
	}
	if w.Body.String() != "png-bytes" {
		t.Errorf("body = %q, want png-bytes", w.Body.String())
	}

	// Unknown id → 404.
	req, _ = http.NewRequest(http.MethodGet, "/v1/images/files/img_unknown", nil)
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Errorf("unknown id status = %d, want 404", w.Code)
	}

	// Expired entry → 404.
	s.imageFiles.mu.Lock()
	s.imageFiles.files[id].expiresAt = time.Now().Add(-time.Minute)
	s.imageFiles.mu.Unlock()
	req, _ = http.NewRequest(http.MethodGet, "/v1/images/files/"+id, nil)
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Errorf("expired id status = %d, want 404", w.Code)
	}
}

func TestConvertMaskToSDCPPRejectsOversized(t *testing.T) {
	// A mask whose declared dimensions exceed ImageMaxDimension must be
	// rejected via DecodeConfig before any large allocation (decompression
	// bomb guard). 4100x1 is tiny in practice but over the per-dimension cap.
	img := image.NewRGBA(image.Rect(0, 0, 4100, 1))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	if _, err := ConvertMaskToSDCPP(buf.Bytes()); err == nil {
		t.Fatal("expected error for oversized mask")
	} else if !strings.Contains(err.Error(), "maximum") {
		t.Errorf("error = %q, want mention of maximum", err.Error())
	}
}

func TestTranscodeOutputImage(t *testing.T) {
	src := tinyPNG(t)

	// PNG passthrough.
	out, ct, err := transcodeOutputImage(t.Context(), src, "png", nil)
	if err != nil {
		t.Fatal(err)
	}
	if ct != "image/png" || !bytes.Equal(out, src) {
		t.Errorf("png passthrough altered bytes (ct=%q)", ct)
	}

	// JPEG transcode.
	q := 80
	out, ct, err = transcodeOutputImage(t.Context(), src, "jpeg", &q)
	if err != nil {
		t.Fatal(err)
	}
	if ct != "image/jpeg" {
		t.Errorf("jpeg content-type = %q", ct)
	}
	if _, _, err := image.Decode(bytes.NewReader(out)); err != nil {
		t.Errorf("jpeg output not decodable: %v", err)
	}

	// WebP transcode (requires ffmpeg; skip when unavailable).
	if webpEncodeAvailable() {
		out, ct, err = transcodeOutputImage(t.Context(), src, "webp", nil)
		if err != nil {
			t.Fatal(err)
		}
		if ct != "image/webp" {
			t.Errorf("webp content-type = %q", ct)
		}
		if len(out) < 12 || string(out[:4]) != "RIFF" || string(out[8:12]) != "WEBP" {
			t.Errorf("output is not a WebP RIFF container: %d bytes", len(out))
		}
	} else {
		t.Log("ffmpeg not available; skipping webp transcode check")
	}

	// Unsupported format.
	if _, _, err := transcodeOutputImage(t.Context(), src, "tiff", nil); err == nil {
		t.Error("expected error for unsupported format")
	}
}
