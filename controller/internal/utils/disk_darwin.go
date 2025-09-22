package utils

import "golang.org/x/sys/unix"

// DiskUsage returns total and free bytes for the filesystem containing path.
// On BSD/Darwin, Bsize is the fundamental block size; Bavail excludes root-reserved.
func DiskUsage(path string) (uint64, uint64, error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return 0, 0, err
	}
	bsize := uint64(st.Bsize)
	total := st.Blocks * bsize
	free := st.Bavail * bsize
	return total, free, nil
}
