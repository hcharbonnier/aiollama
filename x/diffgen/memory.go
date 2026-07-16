package diffgen

import (
	"encoding/json"
	"strings"

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
	"WanVideoPipeline":    true,
	"WanT2VPipeline":      true,
	"WanI2VPipeline":      true,
	"WanFLF2VPipeline":    true,
	"WanVACEPipeline":     true,
	"WanTI2VPipeline":     true,
	"LTXVideoPipeline":    true,
	"LingBotVideoPipeline": true,
}

// DetectModelType reads model_index.json and returns "video" or "image" based
// on the architecture field. Returns "image" as the default when the
// architecture is unknown.
func DetectModelType(modelIndexData []byte) string {
	var index struct {
		Architecture string   `json:"architecture"`
		ClassName    string   `json:"_class_name"`
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(modelIndexData, &index); err != nil {
		return "image"
	}

	arch := index.Architecture
	if arch == "" {
		arch = index.ClassName
	}
	if videoArchitectures[arch] {
		return "video"
	}
	for _, c := range index.Capabilities {
		if c == "video" {
			return "video"
		}
	}
	return "image"
}
