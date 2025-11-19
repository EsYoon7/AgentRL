package snapshots

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/sasha-s/go-deadlock"
	"github.com/thudm/agentrl/controller/internal/utils"
	"go.uber.org/zap"
)

type Store struct {
	logger         *zap.SugaredLogger
	dataMu         map[string]*deadlock.RWMutex
	cacheMu        deadlock.RWMutex
	cacheATimes    map[string]time.Time
	cacheLastPrune time.Time
	tmpMu          deadlock.Mutex
	tmpAllocSizes  map[string]uint64
	RootPath       string
	DataPath       string
	CachePath      string
	TmpPath        string
	NodeIDPath     string
	CapacityLimit  int // percentage of disk space to keep free
}

func NewStore(logger *zap.SugaredLogger, root string) *Store {
	abs, err := filepath.Abs(root)
	if err != nil {
		logger.Errorf("failed to get store path: %v", err)
		abs = root
	}

	s := &Store{
		logger:        logger,
		dataMu:        make(map[string]*deadlock.RWMutex),
		cacheATimes:   make(map[string]time.Time),
		tmpAllocSizes: make(map[string]uint64),
		RootPath:      abs,
		DataPath:      path.Join(abs, "data"),
		CachePath:     path.Join(abs, "cache"),
		TmpPath:       path.Join(abs, "tmp"),
		NodeIDPath:    path.Join(abs, "node_id.txt"),
		CapacityLimit: 5, // always keep at least 5% of disk space frees
	}

	if err = os.MkdirAll(s.DataPath, 0755); err != nil {
		s.logger.Fatalf("failed to initialize store data directory: %v", err)
	}

	if err = os.MkdirAll(s.CachePath, 0755); err != nil {
		s.logger.Fatalf("failed to initialize store cache directory: %v", err)
	}

	if err = os.RemoveAll(s.TmpPath); err != nil {
		s.logger.Fatalf("failed to clear store tmp directory: %v", err)
	}

	if err = os.MkdirAll(s.TmpPath, 0755); err != nil {
		s.logger.Fatalf("failed to initialize store tmp directory: %v", err)
	}

	logger.Infof("initialized store at %s", abs)

	return s
}

// GetNodeID get a locally persisted stable Node ID.
// Currently not used.
func (s *Store) GetNodeID() string {
	data, err := os.ReadFile(s.NodeIDPath)
	if err != nil {
		s.logger.Warnf("failed to read node ID: %v", err)
	}

	id := string(data)
	if id != "" {
		return id
	}

	id, err = os.Hostname()
	if err != nil {
		s.logger.Errorf("failed to get hostname: %v", err)
		id = uuid.New().String()
	}

	if err = os.WriteFile(s.NodeIDPath, []byte(id), 0644); err != nil {
		s.logger.Errorf("failed to write node ID: %v", err)
	}

	return id
}

// CheckData checks if a snapshot with the given ID exists in the data or the cache.
// Return the path if it exists, otherwise returns an empty string.
func (s *Store) CheckData(id string) string {
	dataPath := path.Join(s.DataPath, id)

	if stat, err := os.Stat(dataPath); err == nil && stat.IsDir() {
		return dataPath
	}

	cachePath := path.Join(s.CachePath, id)

	s.cacheMu.Lock()
	defer s.cacheMu.Unlock()

	if stat, err := os.Stat(cachePath); err == nil && stat.IsDir() {
		s.cacheATimes[id] = time.Now()
		return cachePath
	}

	return ""
}

// CreateTemporary creates a new temporary directory for building a snapshot.
// It checks if there is enough disk space available, considering the expected size of the snapshot.
// Returns the path to the temporary directory.
func (s *Store) CreateTemporary(id string, expectedSize uint64) (string, error) {
	if expectedSize == 0 {
		expectedSize = 4 << 30 // defaults to 4GB
	}

	capacity, free, err := utils.DiskUsage(s.RootPath)
	if err != nil {
		return "", fmt.Errorf("failed to get disk capacity: %w", err)
	}

	limit := capacity * uint64(100-s.CapacityLimit) / 100
	used := capacity - free + expectedSize
	for _, size := range s.tmpAllocSizes {
		used += size
	}
	if used > limit {
		// try to prune cache (no more frequently than once per hour)
		if s.cacheLastPrune.IsZero() || time.Since(s.cacheLastPrune) > time.Hour {
			// prevent parallel prune attempts
			s.cacheMu.Lock()
			s.cacheLastPrune = time.Now()
			s.cacheMu.Unlock()

			// try to free enough space + 10GB buffer
			target := used - limit + (10 << 30)
			freed, err := s.PruneCache(target)

			if err != nil {
				s.logger.Errorf("failed to prune cache: %v", err)
			} else if freed > 0 {
				free += freed
				used -= freed
				s.logger.Infof("pruned %d bytes of cache to free up space", freed)
			}
		}

		// recheck space limit
		if used > limit {
			return "", fmt.Errorf("not enough disk space: need %d bytes, have %d bytes free (limit %d%% of %d bytes)", expectedSize, free, s.CapacityLimit, capacity)
		}
	}

	s.tmpMu.Lock()
	defer s.tmpMu.Unlock()

	tmpPath := path.Join(s.TmpPath, id)
	if err = os.MkdirAll(tmpPath, 0755); err != nil {
		return "", err
	}

	s.tmpAllocSizes[id] = expectedSize

	s.logger.Debugw("created new temporary directory", "path", tmpPath, "size", expectedSize)
	return tmpPath, nil
}

// SaveTemporary atomically moves the temporary snapshot directory to the data directory.
// Returns the size of the saved snapshot in bytes.
func (s *Store) SaveTemporary(id string, isCache bool) (uint64, error) {
	tmpPath := path.Join(s.TmpPath, id)

	var finalPath string
	if isCache {
		finalPath = path.Join(s.CachePath, id)
	} else {
		finalPath = path.Join(s.DataPath, id)
		mu, ok := s.dataMu[id]
		if !ok {
			mu = &deadlock.RWMutex{}
			s.dataMu[id] = mu
		}
		mu.Lock()
		defer mu.Unlock()
	}

	// ensure temporary directory exists
	if _, err := os.Stat(tmpPath); os.IsNotExist(err) {
		return 0, fmt.Errorf("temporary snapshot %s does not exist", id)
	}

	// ensure final directory does not exist
	if _, err := os.Stat(finalPath); err == nil {
		s.logger.Warnf("final snapshot path %s already exists, overwriting", finalPath)
		if err = os.RemoveAll(finalPath); err != nil {
			return 0, fmt.Errorf("failed to remove existing snapshot at %s: %w", finalPath, err)
		}
	}

	var size uint64
	err := filepath.Walk(tmpPath, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			s.logger.Warnf("failed to get file info: %v", err)
			return nil
		}
		if !info.IsDir() {
			size += uint64(info.Size())
		}
		return nil
	})
	if err != nil {
		s.logger.Warnf("failed to calculate snapshot size: %v", err)
	}

	if size == 0 {
		return 0, fmt.Errorf("temporary snapshot %s is empty", id)
	}

	if err = os.Rename(tmpPath, finalPath); err != nil {
		return 0, err
	}

	if isCache {
		s.cacheMu.Lock()
		s.cacheATimes[id] = time.Now()
		s.cacheMu.Unlock()
	}

	s.tmpMu.Lock()
	defer s.tmpMu.Unlock()
	delete(s.tmpAllocSizes, id)

	return size, nil
}

// PruneCache removes least recently used snapshots from the cache.
// If maxSize is provided, only remove no more than that many bytes of cache.
// Returns the total size of removed cache in bytes.
func (s *Store) PruneCache(maxSize uint64) (uint64, error) {
	if maxSize == 0 {
		maxSize = 100 << 30 // defaults to 100GB
	}

	s.cacheMu.RLock()
	defer s.cacheMu.RUnlock()

	var totalSize uint64
	aTimes := make(map[string]time.Time)
	sizes := make(map[string]uint64)

	// gather access times and sizes
	if err := filepath.Walk(s.CachePath, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			s.logger.Warnf("failed to get file info: %v", err)
			return err
		}

		// find cache id
		// <cache_dir>/<id>/<possibly_more_paths>
		rel, err := filepath.Rel(s.CachePath, path)
		if err != nil {
			s.logger.Warnf("failed to get relative path: %v", err)
			return nil // continue
		}
		if rel == "" || rel == "." {
			// root of cache directory, skip
			return nil
		}
		parts := filepath.SplitList(rel)
		if parts[0] == "." || parts[0] == ".." {
			s.logger.Warnf("invalid relative path %s in cache directory for %s", rel, path)
			return nil // continue
		}
		id := parts[0]

		if len(parts) > 1 {
			// not the root of a cache entry, accumulate size
			_, ok := sizes[id]
			if !ok {
				sizes[id] = 0
			}
			sizes[id] += uint64(info.Size())
		} else if info.IsDir() {
			// check access time of the directory
			aTime, ok := s.cacheATimes[id]
			if !ok || aTime.IsZero() {
				// fallback to fs mtime if presents
				aTime = info.ModTime()
			}

			// if accessed within 1 hour, skip this directory
			if time.Since(aTime) < time.Hour {
				return fs.SkipDir
			}

			aTimes[id] = aTime
		} else {
			// cache directory should not contain files, remove it
			err = os.Remove(path)
			if err != nil {
				s.logger.Warnf("failed to remove unexpected file in cache directory: %v", err)
			}
			totalSize += uint64(info.Size())

			if totalSize >= maxSize {
				return fs.SkipAll
			}
		}

		return nil
	}); err != nil {
		return 0, err
	}

	// sort entries by access time
	type entry struct {
		path  string
		aTime time.Time
		size  uint64
	}
	entries := make([]entry, 0, len(aTimes))
	for id, aTime := range aTimes {
		size, _ := sizes[id]
		entries = append(entries, entry{
			path:  path.Join(s.CachePath, id),
			aTime: aTime,
			size:  size,
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].aTime.Before(entries[j].aTime)
	})

	// remove least recently used entries
	for _, e := range entries {
		if err := os.RemoveAll(e.path); err != nil {
			s.logger.Warnf("failed to remove cache entry %s: %v", e.path, err)
			continue
		}
		s.logger.Infof("removed cache entry %s last accessed at %s", e.path, e.aTime)
		totalSize += e.size

		if totalSize >= maxSize {
			break
		}
	}

	return totalSize, nil
}

// DeleteSnapshot deletes a snapshot from data, cache, and temporary directories.
func (s *Store) DeleteSnapshot(id string) {
	go func() {
		mu, ok := s.dataMu[id]
		if ok && mu != nil {
			mu.Lock()
			defer mu.Unlock()
		}

		dataPath := path.Join(s.DataPath, id)
		if err := os.RemoveAll(dataPath); err != nil {
			s.logger.Errorf("failed to remove snapshot %s from data: %v", id, err)
		}

		delete(s.dataMu, id)
	}()

	go func() {
		s.cacheMu.Lock()
		defer s.cacheMu.Unlock()

		cachePath := path.Join(s.CachePath, id)
		if err := os.RemoveAll(cachePath); err != nil {
			s.logger.Errorf("failed to remove snapshot %s from cache: %v", id, err)
		} else {
			delete(s.cacheATimes, id)
		}
	}()

	go func() {
		s.tmpMu.Lock()
		s.tmpMu.Unlock()

		tmpPath := path.Join(s.TmpPath, id)
		if err := os.RemoveAll(tmpPath); err != nil {
			s.logger.Errorf("failed to remove snapshot %s from tmp: %v", id, err)
		} else {
			delete(s.tmpAllocSizes, id)
		}
	}()
}

// StreamArchive streams the snapshot directory as a tar archive.
// The caller is responsible for closing the returned ReadCloser.
func (s *Store) StreamArchive(ctx context.Context, id string) (io.ReadCloser, error) {
	dataPath := path.Join(s.DataPath, id)

	info, err := os.Stat(dataPath)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("snapshot %s is not a directory", id)
	}

	pr, pw := io.Pipe()

	go func() {
		mu, ok := s.dataMu[id]
		if !ok {
			mu = &deadlock.RWMutex{}
			s.dataMu[id] = mu
		}
		mu.RLock()
		defer mu.RUnlock()

		defer pw.Close()

		tw := tar.NewWriter(pw)
		defer func() {
			if err := tw.Close(); err != nil {
				_ = pw.CloseWithError(err)
			}
		}()

		if err = filepath.WalkDir(dataPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			rel, err := filepath.Rel(dataPath, path)
			if err != nil {
				return err
			}
			if rel == "" || rel == "." {
				// Skip the root's own header; we'll include subdirs and files.
				return nil
			}
			rel = filepath.ToSlash(rel) // tar uses forward slashes

			fi, err := d.Info()
			if err != nil {
				return err
			}

			mode := fi.Mode()
			switch {
			case mode.IsDir():
				header, err := tar.FileInfoHeader(fi, "")
				if err != nil {
					return err
				}
				header.Name = rel
				if !strings.HasSuffix(header.Name, "/") {
					header.Name += "/"
				}
				header.Uid, header.Gid = 0, 0
				header.Uname, header.Gname = "", ""
				if err = tw.WriteHeader(header); err != nil {
					return err
				}
				return nil

			case mode&os.ModeSymlink != 0:
				link, err := os.Readlink(path)
				if err != nil {
					return err
				}
				header, err := tar.FileInfoHeader(fi, link)
				if err != nil {
					return err
				}
				header.Name = rel
				header.Uid, header.Gid = 0, 0
				header.Uname, header.Gname = "", ""
				return tw.WriteHeader(header)

			case mode.IsRegular():
				header, err := tar.FileInfoHeader(fi, "")
				if err != nil {
					return err
				}
				header.Name = rel
				header.Uid, header.Gid = 0, 0
				header.Uname, header.Gname = "", ""
				if err = tw.WriteHeader(header); err != nil {
					return err
				}
				f, err := os.Open(path)
				if err != nil {
					return err
				}
				_, err = io.Copy(tw, f) // stream file payload
				_ = f.Close()
				return err

			default:
				return nil
			}
		}); err != nil {
			_ = pw.CloseWithError(err)
			return
		}
	}()

	return pr, nil
}

// ExtractArchive extracts a tar archive stream into a temporary snapshot directory.
// The temporary directory must be created with CreateTemporary first and be saving with SaveTemporary after use.
// The caller is responsible for closing the reader.
func (s *Store) ExtractArchive(ctx context.Context, id string, r io.Reader) error {
	tmpPath := path.Join(s.TmpPath, id)

	// check for tmpPath exists
	if stat, err := os.Stat(tmpPath); err != nil || !stat.IsDir() {
		return fmt.Errorf("temporary snapshot %s does not exist", tmpPath)
	}

	tr := tar.NewReader(r)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if header == nil || header.Name == "" {
			continue
		}

		// Normalize and validate header path.
		name := tarPath(header.Name)
		target, err := safeJoin(tmpPath, name)
		if err != nil {
			return err
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err = os.MkdirAll(target, tarFsMode(header.FileInfo().Mode(), 0o755)); err != nil {
				return err
			}
			_ = os.Chtimes(target, time.Now(), header.ModTime)

		case tar.TypeReg:
			if err = os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, tarFsMode(header.FileInfo().Mode(), 0o644))
			if err != nil {
				return err
			}
			if _, err = io.Copy(f, tr); err != nil {
				_ = f.Close()
				return err
			}
			_ = f.Chmod(tarFsMode(header.FileInfo().Mode(), 0o644))
			_ = f.Close()
			_ = os.Chtimes(target, time.Now(), header.ModTime)

		case tar.TypeSymlink:
			// Restrict to relative linknames for safety.
			ln := header.Linkname
			if filepath.IsAbs(ln) || strings.HasPrefix(filepath.Clean(ln), "..") {
				return fmt.Errorf("unsafe symlink target %q in %q", ln, name)
			}
			if err = os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			// Remove existing path if present; then create symlink.
			_ = os.RemoveAll(target)
			if err = os.Symlink(ln, target); err != nil {
				return err
			}

		case tar.TypeLink:
			// Hardlink to another name in the archive.
			srcName := tarPath(header.Linkname)
			src, err := safeJoin(tmpPath, srcName)
			if err != nil {
				return err
			}
			if err = os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			_ = os.RemoveAll(target)
			if err = os.Link(src, target); err != nil {
				return err
			}

		default:
			// Ignore other types (fifo, char, block, etc.)
		}
	}

	return nil
}

func tarPath(name string) string {
	name = strings.TrimSpace(name)
	name = strings.TrimPrefix(name, "./")
	name = strings.TrimPrefix(name, "/")
	name = filepath.Clean(name)
	return name
}

func tarFsMode(m os.FileMode, def os.FileMode) os.FileMode {
	if m == 0 {
		return def
	}
	// Keep only permission bits; ignore special bits for portability.
	return m.Perm()
}

// safeJoin ensures 'name' stays within base; no traversal outside base.
func safeJoin(base, name string) (string, error) {
	if name == "" || name == "." || strings.HasPrefix(name, "..") {
		return "", fmt.Errorf("invalid path %q", name)
	}

	p := filepath.Join(base, name)
	// Ensure prefix match on path separator boundary.
	baseSep := filepath.Clean(base) + string(os.PathSeparator)
	pSep := filepath.Clean(p) + string(os.PathSeparator)
	if !strings.HasPrefix(pSep, baseSep) {
		return "", fmt.Errorf("path escapes base: %q", name)
	}

	return p, nil
}
