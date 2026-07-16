//go:build sdcpp

package diffgen

import (
	"github.com/ollama/ollama/x/sdcpp"
)

// sdContext is the subset of the sdcpp.Context API used by the diffgen HTTP
// handlers. Abstracting it behind an interface lets the handlers be unit-tested
// with a mock instead of requiring a linked libstable-diffusion and a GPU (see
// docs/video-generation-implementation-plan.md §12.2). The concrete
// *sdcpp.Context satisfies this interface; the adapter is in context_sdcpp.go
// (build tag sdcpp).
type sdContext interface {
	GenerateImage(p sdcpp.ImageGenParams, progress sdcpp.ProgressFunc) ([]sdcpp.Image, error)
	GenerateVideo(p sdcpp.VideoGenParams, progress sdcpp.ProgressFunc) ([]sdcpp.Image, error)
	CancelGeneration(mode sdcpp.CancelMode)
	SupportsImageGeneration() bool
	SupportsVideoGeneration() bool
	Close()
}
