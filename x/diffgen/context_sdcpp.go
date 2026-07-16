//go:build sdcpp

package diffgen

import (
	"github.com/ollama/ollama/x/sdcpp"
)

// Compile-time assertion that *sdcpp.Context satisfies sdContext. The methods
// are defined directly on *sdcpp.Context in x/sdcpp/sdcpp.go, so no wrapper
// type is needed.
var _ sdContext = (*sdcpp.Context)(nil)
