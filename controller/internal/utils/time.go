package utils

import "time"

var formatTimeTz, _ = time.LoadLocation("Asia/Shanghai")

func FormatTime(t time.Time) string {
	if formatTimeTz != nil {
		t = t.In(formatTimeTz)
	}

	return t.Format(time.DateTime)
}
