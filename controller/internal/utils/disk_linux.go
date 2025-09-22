package utils

import "golang.org/x/sys/unix"

// DiskUsage returns total and free bytes for the filesystem containing path.
// On Linux, use Frsize when available; Bavail excludes root-reserved blocks.
func DiskUsage(path string) (uint64, uint64, error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return 0, 0, err
	}
	// Prefer fragment size; fallback to block size if needed.
	bsize := uint64(st.Frsize)
	if bsize == 0 {
		bsize = uint64(st.Bsize)
	}
	total := uint64(st.Blocks) * bsize
	free := uint64(st.Bavail) * bsize
	return total, free, nil
}
