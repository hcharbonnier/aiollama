//go:build sdcpp

// This test file is only compiled when libstable-diffusion is available
// (build tag "sdcpp"), because go test links the entire package against the
// shared library. Run with: go test -tags=sdcpp ./x/sdcpp/

package sdcpp

import (
	"testing"
)

func TestCImageToGoEmpty(t *testing.T) {
	var cImg testSDImage
	out := testCImageToGo(&cImg)
	if out.Width != 0 || out.Height != 0 || out.Channel != 0 {
		t.Fatalf("expected zero image, got %+v", out)
	}
	if out.Data != nil {
		t.Fatalf("expected nil data, got %v", out.Data)
	}
}

func TestGoImageToCRoundTrip(t *testing.T) {
	img := &Image{
		Width:   2,
		Height:  2,
		Channel: 3,
		Data:    []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
	}
	out := testGoImageToCAndBack(img)
	if out.Width != img.Width || out.Height != img.Height || out.Channel != img.Channel {
		t.Fatalf("dimension mismatch: got %+v want %+v", out, img)
	}
	if len(out.Data) != len(img.Data) {
		t.Fatalf("data length mismatch: got %d want %d", len(out.Data), len(img.Data))
	}
	for i := range img.Data {
		if out.Data[i] != img.Data[i] {
			t.Fatalf("data mismatch at %d: got %d want %d", i, out.Data[i], img.Data[i])
		}
	}
}

func TestBoolToInt(t *testing.T) {
	if boolToInt(true) != 1 {
		t.Fatal("boolToInt(true) should be 1")
	}
	if boolToInt(false) != 0 {
		t.Fatal("boolToInt(false) should be 0")
	}
}
