package server

import (
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
)

func CreateGrpcServer() *grpc.Server {
	keepAliveEnforcementOption := grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second,
		PermitWithoutStream: true,
	})

	keepAliveOption := grpc.KeepaliveParams(keepalive.ServerParameters{
		MaxConnectionIdle: 10 * time.Minute,
		Time:              10 * time.Second,
		Timeout:           5 * time.Minute,
	})

	maxSizeOption := grpc.MaxRecvMsgSize(100 << 20) // 100 MB

	return grpc.NewServer(keepAliveEnforcementOption, keepAliveOption, maxSizeOption)
}
