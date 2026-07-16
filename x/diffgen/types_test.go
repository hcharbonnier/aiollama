package diffgen

import (
	"encoding/json"
	"testing"
)

func TestDiffRequestMarshalRoundTrip(t *testing.T) {
	original := DiffRequest{
		Prompt:         "a cat on the moon",
		NegativePrompt: "low quality, blurry",
		Mode:           "video",
		Width:          832,
		Height:         480,
		Steps:          20,
		Seed:           42,
		CFGScale:       6.0,
		Sampler:        "euler",
		OutputFormat:   "webm",
		VideoFrames:    33,
		FPS:            16,
		FlowShift:      3.0,
		Images:         [][]byte{{0x89, 0x50, 0x4e, 0x47}},
		EndImage:       []byte{0x89, 0x50, 0x4e, 0x47},
		Options: &RequestOptions{
			NumPredict:  100,
			Temperature: 0.8,
			TopP:        0.9,
			TopK:        40,
			Stop:        []string{"END"},
		},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded DiffRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if decoded.Prompt != original.Prompt {
		t.Errorf("Prompt = %q, want %q", decoded.Prompt, original.Prompt)
	}
	if decoded.Mode != original.Mode {
		t.Errorf("Mode = %q, want %q", decoded.Mode, original.Mode)
	}
	if decoded.VideoFrames != original.VideoFrames {
		t.Errorf("VideoFrames = %d, want %d", decoded.VideoFrames, original.VideoFrames)
	}
	if decoded.FPS != original.FPS {
		t.Errorf("FPS = %d, want %d", decoded.FPS, original.FPS)
	}
	if decoded.FlowShift != original.FlowShift {
		t.Errorf("FlowShift = %f, want %f", decoded.FlowShift, original.FlowShift)
	}
	if decoded.Options == nil || decoded.Options.NumPredict != original.Options.NumPredict {
		t.Errorf("Options.NumPredict mismatch")
	}
	if len(decoded.Images) != 1 || len(decoded.Images[0]) != len(original.Images[0]) {
		t.Errorf("Images mismatch")
	}
	if len(decoded.EndImage) != len(original.EndImage) {
		t.Errorf("EndImage mismatch")
	}
}

func TestDiffRequestOmitEmpty(t *testing.T) {
	req := DiffRequest{Prompt: "hello"}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("Unmarshal to map failed: %v", err)
	}

	for _, key := range []string{"negative_prompt", "mode", "width", "height", "steps", "video_frames", "fps", "images", "end_image", "options"} {
		if _, ok := m[key]; ok {
			t.Errorf("expected %q to be omitted, but it was present", key)
		}
	}
	if _, ok := m["prompt"]; !ok {
		t.Errorf("expected prompt to be present")
	}
}

func TestDiffResponseMarshalRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		resp DiffResponse
	}{
		{
			name: "progress update",
			resp: DiffResponse{Step: 5, Total: 20, Done: false},
		},
		{
			name: "image result",
			resp: DiffResponse{Image: "iVBORw0KGgo=", Done: true},
		},
		{
			name: "video frame",
			resp: DiffResponse{Frame: 3, Frames: 33, Image: "iVBORw0KGgo=", Done: false},
		},
		{
			name: "video done",
			resp: DiffResponse{Done: true, Frames: 33},
		},
		{
			name: "error",
			resp: DiffResponse{Error: "out of memory", Done: true},
		},
		{
			name: "warning",
			resp: DiffResponse{Warning: "WAN VAE CPU fallback", Done: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.resp)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			var decoded DiffResponse
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}

			if decoded.Done != tt.resp.Done {
				t.Errorf("Done = %v, want %v", decoded.Done, tt.resp.Done)
			}
			if decoded.Step != tt.resp.Step {
				t.Errorf("Step = %d, want %d", decoded.Step, tt.resp.Step)
			}
			if decoded.Frame != tt.resp.Frame {
				t.Errorf("Frame = %d, want %d", decoded.Frame, tt.resp.Frame)
			}
			if decoded.Image != tt.resp.Image {
				t.Errorf("Image mismatch")
			}
			if decoded.Content != tt.resp.Content {
				t.Errorf("Content = %q, want %q", decoded.Content, tt.resp.Content)
			}
			if decoded.Error != tt.resp.Error {
				t.Errorf("Error = %q, want %q", decoded.Error, tt.resp.Error)
			}
			if decoded.Warning != tt.resp.Warning {
				t.Errorf("Warning = %q, want %q", decoded.Warning, tt.resp.Warning)
			}
		})
	}
}

func TestModelModeString(t *testing.T) {
	if ModeImage.String() != "image" {
		t.Errorf("ModeImage.String() = %q, want \"image\"", ModeImage.String())
	}
	if ModeVideo.String() != "video" {
		t.Errorf("ModeVideo.String() = %q, want \"video\"", ModeVideo.String())
	}
}
