package utils

import "golang.org/x/sys/windows"

// DiskUsage returns total and free bytes for the volume containing path.
func DiskUsage(path string) (uint64, uint64, error) {
	p16, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, 0, err
	}
	var freeAvail, total, free uint64
	// freeAvail: available to the *calling user* (quota-aware);
	// free: total free on the volume (not quota-aware). We return freeAvail.
	if err = windows.GetDiskFreeSpaceEx(p16, &freeAvail, &total, &free); err != nil {
		return 0, 0, err
	}
	return total, freeAvail, nil
}
