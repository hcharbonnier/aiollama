//go:build !sdcpp

// This file provides stub implementations for the diffgen entry points when
// the sdcpp build tag is not set (i.e. libstable-diffusion is not linked).
// The scheduler imports NewServer unconditionally; the stub returns a clear
// error so that non-sdcpp builds fail gracefully at runtime if a diffgen
// model is requested. IsDiffModel/ResolveModelName are in untagged detect.go
// and work in both build modes.

package diffgen

import (
	"errors"

	"github.com/ollama/ollama/llm"
)

// errSDCppNotCompiled is returned by all diffgen entry points when the
// binary was not compiled with the sdcpp build tag.
var errSDCppNotCompiled = errors.New("diffgen models require a build with the sdcpp tag (libstable-diffusion not linked)")

// NewServer is a stub that always fails without the sdcpp build tag.
func NewServer(modelName, mode string) (llm.LlamaServer, error) {
	return nil, errSDCppNotCompiled
}

