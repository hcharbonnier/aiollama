package diffgen

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

// CheckPlatformSupport validates that diffusion generation is supported on the
// current platform. SD.cpp supports all platforms (CPU/CUDA/Metal/Vulkan), so
// this always returns nil.
func CheckPlatformSupport() error {
	return nil
}

// ResolveBackend maps discovered GPU devices to an SD.cpp backend string.
// Preference order: CUDA > Metal > Vulkan > CPU. Scans all GPUs so a
// higher-priority backend wins even if it appears later in the list.
func ResolveBackend(gpus []ml.DeviceInfo) string {
	var hasCUDA, hasMetal, hasVulkan bool
	for _, g := range gpus {
		switch strings.ToLower(g.Library) {
		case "cuda":
			hasCUDA = true
		case "metal":
			hasMetal = true
		case "vulkan":
			hasVulkan = true
		}
	}
	switch {
	case hasCUDA:
		return "cuda"
	case hasMetal:
		return "metal"
	case hasVulkan:
		return "vulkan"
	default:
		return "cpu"
	}
}

// videoArchitectures lists model_index.json architecture values that indicate
// a video generation model.
var videoArchitectures = map[string]bool{
	"WanVideoPipeline":     true,
	"WanT2VPipeline":       true,
	"WanI2VPipeline":       true,
	"WanFLF2VPipeline":     true,
	"WanVACEPipeline":      true,
	"WanTI2VPipeline":      true,
	"LTXVideoPipeline":     true,
	"LingBotVideoPipeline": true,
}

// IsWANVideoArchitecture reports whether the architecture is a WAN video
// pipeline subject to the CUDA/CPU-only VAE limitation. WAN pipelines all
// share the "Wan" prefix in their architecture name.
func IsWANVideoArchitecture(arch string) bool {
	return videoArchitectures[arch] && strings.HasPrefix(arch, "Wan")
}

// DetectModelType reads model_index.json and returns "video" or "image" based
// on the architecture field. Returns "image" as the default when the
// architecture is unknown.
func DetectModelType(modelIndexData []byte) string {
	arch := DetectArchitecture(modelIndexData)
	if videoArchitectures[arch] {
		return "video"
	}
	var index struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(modelIndexData, &index); err == nil {
		for _, c := range index.Capabilities {
			if c == "video" {
				return "video"
			}
		}
	}
	return "image"
}

// DetectArchitecture reads model_index.json and returns the architecture (or
// _class_name fallback). Returns "" on parse failure.
func DetectArchitecture(modelIndexData []byte) string {
	var index struct {
		Architecture string `json:"architecture"`
		ClassName    string `json:"_class_name"`
	}
	if err := json.Unmarshal(modelIndexData, &index); err != nil {
		return ""
	}
	if index.Architecture != "" {
		return index.Architecture
	}
	return index.ClassName
}

// EstimateVRAMBudget returns the free VRAM budget (in bytes) available for
// SD.cpp model offload on the given backend, after subtracting the per-GPU
// minimum and global GPU overhead. It sums free memory across all GPUs whose
// library matches the resolved backend, so multi-GPU hosts are correctly
// budgeted. Returns 0 when no matching GPUs are available, signalling SD.cpp
// to run fully on CPU.
func EstimateVRAMBudget(gpus []ml.DeviceInfo, backend string) uint64 {
	if len(gpus) == 0 {
		return 0
	}
	var total uint64
	for _, g := range gpus {
		if strings.ToLower(g.Library) != backend {
			continue
		}
		available := g.FreeMemory
		overhead := g.MinimumMemory() + envconfig.GpuOverhead()
		if available > overhead {
			total += available - overhead
		}
	}
	return total
}

// FormatVRAMGiB converts a byte count to the GiB string SD.cpp expects for its
// max_vram context parameter (e.g. 8589934592 -> "8"). Rounds to the nearest
// GiB to avoid under-provisioning (flooring would lose up to ~1 GiB). Returns
// "" when the budget is zero so the field is left unset (no offload budget).
func FormatVRAMGiB(bytes uint64) string {
	if bytes == 0 {
		return ""
	}
	const gib = uint64(1024 * 1024 * 1024)
	rounded := (bytes + gib/2) / gib
	if rounded == 0 {
		return ""
	}
	return fmt.Sprintf("%d", rounded)
}

// ShouldStreamLayers decides whether to enable SD.cpp layer streaming
// (residency+prefetch). Streaming is only enabled when the model size exceeds
// the available VRAM budget, so models that fit entirely in VRAM don't pay
// the streaming/prefetch overhead. When the budget is zero (CPU-only), no
// streaming is needed.
func ShouldStreamLayers(modelSize, vramBudget uint64) bool {
	if vramBudget == 0 {
		return false
	}
	return modelSize > vramBudget
}

// WANVAEDeprecatedBackend reports whether a WAN video model is being loaded on
// a backend whose VAE does not support CUDA (Metal/Vulkan), requiring a CPU
// VAE fallback.
func WANVAEDeprecatedBackend(architecture, backend string) bool {
	if !IsWANVideoArchitecture(architecture) {
		return false
	}
	switch backend {
	case "cuda", "cpu", "":
		return false
	default:
		return true
	}
}
