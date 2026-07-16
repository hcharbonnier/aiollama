// Package diffutil provides shared CLI utilities for diffusion model
// generation (image and video). These helpers are used by both the
// x/diffgen and x/imagegen CLI packages to avoid code duplication.
package diffutil

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

// SanitizeFilename removes characters that aren't safe for filenames.
func SanitizeFilename(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, " ", "-")
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// DisplayImageInTerminal attempts to render an image inline in the terminal.
// Supports iTerm2, Kitty, WezTerm, Ghostty, and other terminals with inline
// image support. Returns true if the image was displayed, false otherwise.
func DisplayImageInTerminal(imagePath string) bool {
	termProgram := os.Getenv("TERM_PROGRAM")
	kittyWindowID := os.Getenv("KITTY_WINDOW_ID")
	weztermPane := os.Getenv("WEZTERM_PANE")
	ghostty := os.Getenv("GHOSTTY_RESOURCES_DIR")

	data, err := os.ReadFile(imagePath)
	if err != nil {
		return false
	}
	encoded := base64.StdEncoding.EncodeToString(data)

	switch {
	case termProgram == "iTerm.app" || termProgram == "WezTerm" || weztermPane != "":
		fmt.Printf("\033]1337;File=inline=1;preserveAspectRatio=1:%s\a\n", encoded)
		return true
	case kittyWindowID != "" || ghostty != "" || termProgram == "ghostty":
		const chunkSize = 4096
		for i := 0; i < len(encoded); i += chunkSize {
			end := min(i+chunkSize, len(encoded))
			chunk := encoded[i:end]
			if i == 0 {
				more := 1
				if end >= len(encoded) {
					more = 0
				}
				fmt.Printf("\033_Ga=T,f=100,m=%d;%s\033\\", more, chunk)
			} else if end >= len(encoded) {
				fmt.Printf("\033_Gm=0;%s\033\\", chunk)
			} else {
				fmt.Printf("\033_Gm=1;%s\033\\", chunk)
			}
		}
		fmt.Println()
		return true
	default:
		return false
	}
}

// ExtractFileNames finds image file paths in the input string.
func ExtractFileNames(input string) []string {
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp)\b`
	re := regexp.MustCompile(regexPattern)
	return re.FindAllString(input, -1)
}

// ExtractFileData extracts image data from file paths found in the input.
// Returns the cleaned prompt (with file paths removed) and the image data.
func ExtractFileData(input string) (string, []api.ImageData, error) {
	filePaths := ExtractFileNames(input)
	var imgs []api.ImageData
	for _, fp := range filePaths {
		nfp := strings.ReplaceAll(fp, "\\ ", " ")
		nfp = strings.ReplaceAll(nfp, "\\(", "(")
		nfp = strings.ReplaceAll(nfp, "\\)", ")")
		nfp = strings.ReplaceAll(nfp, "%20", " ")
		data, err := GetImageData(nfp)
		if errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return "", nil, err
		}
		fmt.Fprintf(os.Stderr, "Added image '%s'\n", nfp)
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, data)
	}
	return strings.TrimSpace(input), imgs, nil
}

// GetImageData reads and validates image data from a file.
func GetImageData(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	buf := make([]byte, 512)
	if _, err = file.Read(buf); err != nil {
		return nil, err
	}
	contentType := http.DetectContentType(buf)
	allowedTypes := []string{"image/jpeg", "image/jpg", "image/png", "image/webp"}
	if !slices.Contains(allowedTypes, contentType) {
		return nil, fmt.Errorf("invalid image type: %s", contentType)
	}
	return os.ReadFile(filePath)
}
