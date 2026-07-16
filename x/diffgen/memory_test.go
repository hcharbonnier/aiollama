package diffgen

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestDetectModelType(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{
			name: "wan video architecture",
			data: `{"architecture":"WanVideoPipeline"}`,
			want: "video",
		},
		{
			name: "ltx video architecture",
			data: `{"architecture":"LTXVideoPipeline"}`,
			want: "video",
		},
		{
			name: "flux image architecture",
			data: `{"architecture":"FluxPipeline"}`,
			want: "image",
		},
		{
			name: "diffusers class name video",
			data: `{"_class_name":"WanT2VPipeline"}`,
			want: "video",
		},
		{
			name: "capabilities video field",
			data: `{"architecture":"CustomPipeline","capabilities":["video"]}`,
			want: "video",
		},
		{
			name: "unknown architecture defaults to image",
			data: `{"architecture":"SomeUnknownThing"}`,
			want: "image",
		},
		{
			name: "invalid json defaults to image",
			data: `{not json`,
			want: "image",
		},
		{
			name: "empty defaults to image",
			data: `{}`,
			want: "image",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectModelType([]byte(tt.data))
			if got != tt.want {
				t.Fatalf("DetectModelType(%q) = %q, want %q", tt.data, got, tt.want)
			}
		})
	}
}

func TestResolveBackend(t *testing.T) {
	tests := []struct {
		name  string
		libs  []string
		want  string
	}{
		{"cuda preferred", []string{"cpu", "cuda"}, "cuda"},
		{"metal preferred over vulkan", []string{"vulkan", "metal"}, "metal"},
		{"vulkan when no cuda/metal", []string{"vulkan", "cpu"}, "vulkan"},
		{"cpu fallback when empty", []string{}, "cpu"},
		{"cpu fallback when unknown", []string{"unknown_lib"}, "cpu"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gpus := make([]ml.DeviceInfo, len(tt.libs))
			for i, lib := range tt.libs {
				gpus[i].Library = lib
			}
			got := ResolveBackend(gpus)
			if got != tt.want {
				t.Fatalf("ResolveBackend() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCheckPlatformSupportAlwaysNil(t *testing.T) {
	if err := CheckPlatformSupport(); err != nil {
		t.Fatalf("CheckPlatformSupport() = %v, want nil", err)
	}
}
