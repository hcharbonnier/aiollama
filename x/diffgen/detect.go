package diffgen

import (
	"github.com/ollama/ollama/x/diffgen/manifest"
)

// loadManifestForCheck loads the manifest for a model name and returns it if
// the model has a diffusion component (diffusion_model or unet). This is used
// by IsDiffModel to decide whether the diffgen CLI should handle a given
// model, avoiding misrouting ordinary LLM models that also have manifests.
func loadManifestForCheck(modelName string) (*manifest.ModelManifest, error) {
	m, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, err
	}
	if !m.HasComponent("diffusion_model") && !m.HasComponent("unet") {
		return nil, errNotDiffModel
	}
	return m, nil
}

// errNotDiffModel is returned when a manifest exists but has no diffusion
// component, indicating it is not a diffgen model.
var errNotDiffModel = &notDiffModelError{}

type notDiffModelError struct{}

func (e *notDiffModelError) Error() string { return "model has no diffusion component" }
