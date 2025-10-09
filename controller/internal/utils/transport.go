package utils

import (
	"net"
	"net/http"
	"time"
)

func NewDialer() *net.Dialer {
	return &net.Dialer{
		Timeout:   10 * time.Minute,
		KeepAlive: 10 * time.Second,
	}
}

func NewTransport() *http.Transport {
	return &http.Transport{
		IdleConnTimeout:       10 * time.Minute,
		ResponseHeaderTimeout: 5 * time.Minute,
		ExpectContinueTimeout: 5 * time.Minute,
		DialContext:           NewDialer().DialContext,
	}
}
