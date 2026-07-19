package server

import (
	"container/list"
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
	// element is the entry's position in the store's insertion-order list
	// (value: the image id).
	element *list.Element
}

// imageFileStore is a small in-memory store backing
// response_format=url on the OpenAI Images API. The OpenAI cloud API
// returns short-lived signed URLs; aiollama serves the bytes itself from
// GET /v1/images/files/{id} until the TTL expires. The store is
// process-local (lost on restart), like the video job store. A background
// sweep reclaims expired entries every minute (mirroring videojobs'
// evictLoop) so idle servers don't retain a full store indefinitely. All
// entries share one TTL, so insertion order is expiry order: the oldest
// entry is always at the front of the list, making cap eviction O(1) per
// evicted entry instead of a full map scan.
type imageFileStore struct {
	mu         sync.Mutex
	files      map[string]*storedImage
	oldest     *list.List // ids in insertion order; front = oldest
	totalBytes int64
	done       chan struct{}
	closeOnce  sync.Once
}

func newImageFileStore() *imageFileStore {
	s := &imageFileStore{
		files:  make(map[string]*storedImage),
		oldest: list.New(),
		done:   make(chan struct{}),
	}
	go s.sweepLoop()
	return s
}

// Close stops the background sweep.
func (s *imageFileStore) Close() {
	s.closeOnce.Do(func() { close(s.done) })
}

// sweepLoop periodically purges expired entries.
func (s *imageFileStore) sweepLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case now := <-ticker.C:
			s.mu.Lock()
			s.evictExpiredLocked(now)
			s.mu.Unlock()
		}
	}
}

// newImageFileID generates a spec-plausible image file id: "img_" + 24 hex.
func newImageFileID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "img_" + hex.EncodeToString(b)
}

// put stores the image and returns its id. The global byte cap is enforced
// by evicting oldest-first (front of the insertion-order list, O(1) per
// evicted entry). Expired entries are reclaimed by the 1-minute sweep and
// lazily on get, so no per-put expiry scan is needed.
func (s *imageFileStore) put(data []byte, contentType string) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := newImageFileID()
	img := &storedImage{
		data:        data,
		contentType: contentType,
		expiresAt:   time.Now().Add(imageFileTTL),
	}
	img.element = s.oldest.PushBack(id)
	s.files[id] = img
	s.totalBytes += int64(len(data))

	// Evict oldest-first until under the cap, keeping the just-stored entry.
	for s.totalBytes > maxImageFileBytes && len(s.files) > 1 {
		el := s.oldest.Front()
		if el == nil {
			break
		}
		s.evictLocked(el.Value.(string))
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
		s.evictLocked(id)
		return nil, "", false
	}
	return f.data, f.contentType, true
}

// evictLocked removes a single entry. Caller holds s.mu.
func (s *imageFileStore) evictLocked(id string) {
	f, ok := s.files[id]
	if !ok {
		return
	}
	s.totalBytes -= int64(len(f.data))
	s.oldest.Remove(f.element)
	delete(s.files, id)
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
		s.evictLocked(id)
	}
}
