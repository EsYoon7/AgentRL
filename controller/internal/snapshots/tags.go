package snapshots

import (
	"sort"
	"strings"
)

// normalizeTags trims whitespace, removes empty entries, and deduplicates in a stable order.
func normalizeTags(tags []string) []string {
	if len(tags) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(tags))
	var normalized []string
	for _, tag := range tags {
		tag = strings.TrimSpace(tag)
		if tag == "" {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		normalized = append(normalized, tag)
	}
	if len(normalized) == 0 {
		return nil
	}
	sort.Strings(normalized)
	return normalized
}
