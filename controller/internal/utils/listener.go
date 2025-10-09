package utils

import (
	"net"
	"time"
)

// KeepAliveListener wraps the TCP listener to activate keep-alive on accepted connections.
// This hopefully prevents e.g. Docker from closing connections that takes long time to complete.
type KeepAliveListener struct {
	*net.TCPListener
}

func (ln KeepAliveListener) Accept() (net.Conn, error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return nil, err
	}
	err = tc.SetKeepAlive(true)
	if err != nil {
		return nil, err
	}
	err = tc.SetKeepAlivePeriod(10 * time.Second)
	if err != nil {
		return nil, err
	}
	return tc, nil
}
