package utils

import "math/rand"

func RandInt(min int, max int) int {
	if min > max {
		min, max = max, min // swap to ensure min <= max
	}
	return rand.Intn(max-min+1) + min
}
