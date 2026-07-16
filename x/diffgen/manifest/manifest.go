// Package manifest defines the component-file manifest format used by diffgen
// models. Unlike the per-tensor MLX manifest, SD.cpp reads whole checkpoint
// files, so each component (diffusion_model, vae, t5xxl, clip_vision) is stored
// as a single content-addressed blob.
package manifest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// ComponentLayer represents a single model component blob in the manifest.
type ComponentLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	Name      string `json:"name,omitempty"` // e.g. "diffusion_model", "vae", "t5xxl"
}

// Manifest is the on-disk manifest JSON structure.
type Manifest struct {
	SchemaVersion int              `json:"schemaVersion"`
	Config        ComponentLayer   `json:"config"`
	Layers        []ComponentLayer `json:"layers"`
}

// ModelManifest holds a parsed manifest with helper accessors.
type ModelManifest struct {
	Manifest *Manifest
	BlobDir  string
}

// DefaultBlobDir returns the blob storage directory.
func DefaultBlobDir() string {
	return filepath.Join(envconfig.Models(), "blobs")
}

// DefaultManifestDir returns the manifest storage directory.
func DefaultManifestDir() string {
	return filepath.Join(envconfig.Models(), "manifests")
}

// LoadManifest loads a manifest for the given model name.
func LoadManifest(modelName string) (*ModelManifest, error) {
	manifestPath := resolveManifestPath(modelName)
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	return &ModelManifest{Manifest: &m, BlobDir: DefaultBlobDir()}, nil
}

func resolveManifestPath(modelName string) string {
	host := "registry.ollama.ai"
	namespace := "library"
	name := modelName
	tag := "latest"
	if idx := strings.LastIndex(name, ":"); idx != -1 {
		tag = name[idx+1:]
		name = name[:idx]
	}
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 3:
		host = parts[0]
		namespace = parts[1]
		name = parts[2]
	case 2:
		namespace = parts[0]
		name = parts[1]
	}
	return filepath.Join(DefaultManifestDir(), host, namespace, name, tag)
}

// BlobPath returns the full path to a blob given its digest.
func (m *ModelManifest) BlobPath(digest string) string {
	blobName := strings.Replace(digest, ":", "-", 1)
	return filepath.Join(m.BlobDir, blobName)
}

// ComponentPath returns the filesystem path to a named component blob.
func (m *ModelManifest) ComponentPath(name string) (string, error) {
	for _, layer := range m.Manifest.Layers {
		if layer.Name == name {
			return m.BlobPath(layer.Digest), nil
		}
	}
	return "", fmt.Errorf("component %q not found in manifest", name)
}

// ReadConfig reads and returns the config blob content (model_index.json).
func (m *ModelManifest) ReadConfig() ([]byte, error) {
	return os.ReadFile(m.BlobPath(m.Manifest.Config.Digest))
}

// TotalComponentSize returns the sum of all component layer sizes.
func (m *ModelManifest) TotalComponentSize() int64 {
	var total int64
	for _, layer := range m.Manifest.Layers {
		total += layer.Size
	}
	return total
}

// HasComponent reports whether a named component exists in the manifest.
func (m *ModelManifest) HasComponent(name string) bool {
	for _, layer := range m.Manifest.Layers {
		if layer.Name == name {
			return true
		}
	}
	return false
}
