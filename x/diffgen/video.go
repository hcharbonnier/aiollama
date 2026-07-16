//go:build sdcpp

package diffgen

// Video container encoding helpers live here. Phase 1 streams PNG frames
// individually via ndjson in handleVideoCompletion (runner.go), so no
// container encoder is needed yet.
//
// Future phases will add WebM/animated-WebP/GIF container encoding behind
// build tags, wired into the non-streaming response path.
