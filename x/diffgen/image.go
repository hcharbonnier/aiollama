//go:build sdcpp

package diffgen

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"

	"github.com/ollama/ollama/x/sdcpp"
)

// EncodeImageBase64 encodes an sdcpp.Image (raw RGB) as base64-encoded PNG.
func EncodeImageBase64(img sdcpp.Image) (string, error) {
	rgba, err := ImageToRGBA(img)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, rgba); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// ImageToRGBA converts an sdcpp.Image (channel-first raw data) to image.RGBA.
func ImageToRGBA(img sdcpp.Image) (*image.RGBA, error) {
	if img.Channel != 3 {
		return nil, fmt.Errorf("expected 3 channels (RGB), got %d", img.Channel)
	}
	w, h := img.Width, img.Height
	expected := w * h * img.Channel
	if len(img.Data) < expected {
		return nil, fmt.Errorf("image data too short: got %d bytes, need %d", len(img.Data), expected)
	}

	goImg := image.NewRGBA(image.Rect(0, 0, w, h))
	pix := goImg.Pix
	for y := range h {
		for x := range w {
			srcIdx := (y*w + x) * img.Channel
			dstIdx := (y*w + x) * 4
			pix[dstIdx+0] = img.Data[srcIdx+0]
			pix[dstIdx+1] = img.Data[srcIdx+1]
			pix[dstIdx+2] = img.Data[srcIdx+2]
			pix[dstIdx+3] = 255
		}
	}
	return goImg, nil
}

// DecodeImage decodes image bytes, flattening transparency onto white.
func DecodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return flattenAlpha(img), nil
}

// ImageToSDImage converts a Go image.Image to an sdcpp.Image (raw RGB).
func ImageToSDImage(img image.Image) (sdcpp.Image, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)

	data := make([]byte, w*h*3)
	for y := range h {
		for x := range w {
			srcIdx := (y*w + x) * 4
			dstIdx := (y*w + x) * 3
			data[dstIdx+0] = rgba.Pix[srcIdx+0]
			data[dstIdx+1] = rgba.Pix[srcIdx+1]
			data[dstIdx+2] = rgba.Pix[srcIdx+2]
		}
	}
	return sdcpp.Image{Width: w, Height: h, Channel: 3, Data: data}, nil
}

// flattenAlpha composites an image onto a white background, removing alpha.
func flattenAlpha(img image.Image) image.Image {
	if _, ok := img.(*image.RGBA); !ok {
		if _, ok := img.(*image.NRGBA); !ok {
			return img
		}
	}
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, &image.Uniform{color.White}, image.Point{}, draw.Src)
	draw.Draw(dst, bounds, img, bounds.Min, draw.Over)
	return dst
}
