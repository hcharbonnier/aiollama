package server

import (
	"crypto/rand"
	"encoding/hex"
	"sort"
	"sync"
	"time"
)

// imageFileTTL is how long a generated image is retained for
// response_format=url downloads. Mirrors videojobs.JobTTL: clients should
// download promptly; URLs are not permanent storage.
const imageFileTTL = 30 * time.Minute

// maxImageFileBytes bounds the aggregate retained image bytes across the
// store. When exceeded, the oldest entries are evicted (LRU).
const maxImageFileBytes int64 = 512 << 20 // 512 MiB

// storedImage is a single retained generated image.
type storedImage struct {
	data        []byte
	contentType string
	expiresAt   time.Time
}

// imageFileStore is a small in-memory store backing
// response_format=url on the OpenAI Images API. The OpenAI cloud API
// returns short-lived signed URLs; aiollama serves the bytes itself from
// GET /v1/images/files/{id} until the TTL expires. The store is
// process-local (lost on restart), like the video job store.
type imageFileStore struct {
	mu         sync.Mutex
	files      map[string]*storedImage
	totalBytes int64
}

func newImageFileStore() *imageFileStore {
	return &imageFileStore{files: make(map[string]*storedImage)}
}

// newImageFileID generates a spec-plausible image file id: "img_" + 24 hex.
func newImageFileID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "img_" + hex.EncodeToString(b)
}

// put stores the image and returns its id. Expired entries are purged and
// the global byte cap is enforced by evicting oldest-first.
func (s *imageFileStore) put(data []byte, contentType string) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.evictExpiredLocked(time.Now())

	id := newImageFileID()
	s.files[id] = &storedImage{
		data:        data,
		contentType: contentType,
		expiresAt:   time.Now().Add(imageFileTTL),
	}
	s.totalBytes += int64(len(data))

	// LRU-evict oldest until under the cap.
	for s.totalBytes > maxImageFileBytes && len(s.files) > 1 {
		oldestID := ""
		var oldestExp time.Time
		for fid, f := range s.files {
			if fid == id {
				continue
			}
			if oldestID == "" || f.expiresAt.Before(oldestExp) {
				oldestID, oldestExp = fid, f.expiresAt
			}
		}
		if oldestID == "" {
			break
		}
		s.totalBytes -= int64(len(s.files[oldestID].data))
		delete(s.files, oldestID)
	}
	return id
}

// get returns the stored image bytes and content type, or false if the id
// is unknown or expired. Expired entries are removed lazily.
func (s *imageFileStore) get(id string) ([]byte, string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, ok := s.files[id]
	if !ok {
		return nil, "", false
	}
	if time.Now().After(f.expiresAt) {
		s.totalBytes -= int64(len(f.data))
		delete(s.files, id)
		return nil, "", false
	}
	return f.data, f.contentType, true
}

// evictExpiredLocked removes all expired entries. Caller holds s.mu.
func (s *imageFileStore) evictExpiredLocked(now time.Time) {
	// Collect ids first to avoid mutating during iteration.
	expired := make([]string, 0)
	for id, f := range s.files {
		if now.After(f.expiresAt) {
			expired = append(expired, id)
		}
	}
	sort.Strings(expired) // deterministic for tests
	for _, id := range expired {
		s.totalBytes -= int64(len(s.files[id].data))
		delete(s.files, id)
	}
}
