package utils

import (
	"reflect"

	"github.com/thudm/agentrl/controller/internal/types"
)

func ContainsTaskIndex(indices []types.TaskIndex, index types.TaskIndex) bool {
	for _, i := range indices {
		if i.Equals(index) {
			return true
		}
	}
	return false
}

func CompareTaskIndices(a, b []types.TaskIndex) bool {
	return reflect.DeepEqual(a, b)
}
