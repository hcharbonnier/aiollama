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
		name string
		libs []string
		want string
	}{
		{"cuda preferred", []string{"cpu", "cuda"}, "cuda"},
		{"rocm selected", []string{"cpu", "ROCm"}, "rocm"},
		{"cuda preferred over rocm", []string{"ROCm", "cuda"}, "cuda"},
		{"rocm preferred over vulkan", []string{"vulkan", "ROCm"}, "rocm"},
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

func TestDetectArchitecture(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{"architecture field", `{"architecture":"WanVideoPipeline"}`, "WanVideoPipeline"},
		{"class name fallback", `{"_class_name":"FluxPipeline"}`, "FluxPipeline"},
		{"architecture preferred over class", `{"architecture":"SD3Pipeline","_class_name":"FluxPipeline"}`, "SD3Pipeline"},
		{"empty json", `{}`, ""},
		{"invalid json", `{not json`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectArchitecture([]byte(tt.data))
			if got != tt.want {
				t.Fatalf("DetectArchitecture(%q) = %q, want %q", tt.data, got, tt.want)
			}
		})
	}
}

func TestIsWANVideoArchitecture(t *testing.T) {
	tests := []struct {
		arch string
		want bool
	}{
		{"WanVideoPipeline", true},
		{"WanT2VPipeline", true},
		{"WanI2VPipeline", true},
		{"WanFLF2VPipeline", true},
		{"WanVACEPipeline", true},
		{"WanTI2VPipeline", true},
		{"LTXVideoPipeline", false},
		{"FluxPipeline", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			if got := IsWANVideoArchitecture(tt.arch); got != tt.want {
				t.Fatalf("IsWANVideoArchitecture(%q) = %v, want %v", tt.arch, got, tt.want)
			}
		})
	}
}

func TestWANVAEDeprecatedBackend(t *testing.T) {
	tests := []struct {
		name    string
		arch    string
		backend string
		want    bool
	}{
		{"wan on metal", "WanVideoPipeline", "metal", true},
		{"wan on vulkan", "WanT2VPipeline", "vulkan", true},
		{"wan on cuda", "WanVideoPipeline", "cuda", false},
		{"wan on rocm", "WanVideoPipeline", "rocm", false},
		{"wan on cpu", "WanVideoPipeline", "cpu", false},
		{"wan on empty backend", "WanVideoPipeline", "", false},
		{"non-wan on metal", "LTXVideoPipeline", "metal", false},
		{"non-wan on vulkan", "FluxPipeline", "vulkan", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := WANVAEDeprecatedBackend(tt.arch, tt.backend); got != tt.want {
				t.Fatalf("WANVAEDeprecatedBackend(%q, %q) = %v, want %v", tt.arch, tt.backend, got, tt.want)
			}
		})
	}
}

func TestEstimateVRAMBudget(t *testing.T) {
	cudaGPU := func(free uint64) ml.DeviceInfo {
		return ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "cuda"}, FreeMemory: free}
	}
	tests := []struct {
		name     string
		gpus     []ml.DeviceInfo
		backend  string
		wantZero bool
	}{
		{"no gpus", []ml.DeviceInfo{}, "cpu", true},
		{"gpu with zero free", []ml.DeviceInfo{cudaGPU(0)}, "cuda", true},
		{"non-matching backend", []ml.DeviceInfo{cudaGPU(8 << 30)}, "metal", true},
		{"matching backend sums", []ml.DeviceInfo{cudaGPU(4 << 30), cudaGPU(4 << 30)}, "cuda", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EstimateVRAMBudget(tt.gpus, tt.backend)
			if tt.wantZero && got != 0 {
				t.Fatalf("EstimateVRAMBudget() = %d, want 0", got)
			}
			if !tt.wantZero && got == 0 {
				t.Fatalf("EstimateVRAMBudget() = 0, want non-zero")
			}
		})
	}
}

func TestFormatVRAMGiB(t *testing.T) {
	gib := uint64(1024 * 1024 * 1024)
	tests := []struct {
		bytes uint64
		want  string
	}{
		{0, ""},
		{1, ""},
		{gib, "1"},
		{8 * gib, "8"},
		{8*gib + gib/2, "9"},         // rounds up at 0.5
		{8*gib + gib/2 - 1, "8"},     // rounds down just below 0.5
		{8*gib + 512*1024*1024, "9"}, // 8.5 GiB rounds to 9
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := FormatVRAMGiB(tt.bytes)
			if got != tt.want {
				t.Fatalf("FormatVRAMGiB(%d) = %q, want %q", tt.bytes, got, tt.want)
			}
		})
	}
}

func TestShouldStreamLayers(t *testing.T) {
	gib := uint64(1024 * 1024 * 1024)
	if ShouldStreamLayers(4*gib, 0) {
		t.Fatal("ShouldStreamLayers with zero budget = true, want false")
	}
	if ShouldStreamLayers(4*gib, 8*gib) {
		t.Fatal("ShouldStreamLayers with model < budget = true, want false")
	}
	if !ShouldStreamLayers(8*gib, 4*gib) {
		t.Fatal("ShouldStreamLayers with model > budget = false, want true")
	}
	if ShouldStreamLayers(8*gib, 8*gib) {
		t.Fatal("ShouldStreamLayers with model == budget = true, want false (model fits)")
	}
}
