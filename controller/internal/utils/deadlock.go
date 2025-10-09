package utils

import "github.com/sasha-s/go-deadlock"

func ConfigureDeadlock(debug bool) {
	deadlock.Opts.Disable = !debug
}
