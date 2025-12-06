package server

import (
	"context"
	"math"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

type GrpcOptions struct {
	Logger            *zap.SugaredLogger
	LongTimeout       bool
	LargeFile         bool
	DisableReflection bool
}

func NewGrpcServer(op GrpcOptions) *grpc.Server {
	options := []grpc.ServerOption{
		grpc.MaxConcurrentStreams(math.MaxUint32),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             5 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.MaxRecvMsgSize(100 << 20), // 100 MB
	}

	if op.Logger != nil {
		options = append(options,
			grpc.UnaryInterceptor(func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
				resp, err = handler(ctx, req)
				if err != nil {
					st, _ := status.FromError(err)
					op.Logger.Errorw("grpc error", "method", info.FullMethod, "code", st.Code(), "message", st.Message())
				}
				return resp, err
			}),
			grpc.StreamInterceptor(func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
				err := handler(srv, ss)
				if err != nil {
					st, _ := status.FromError(err)
					op.Logger.Errorw("grpc error", "method", info.FullMethod, "code", st.Code(), "message", st.Message())
				}
				return err
			}),
		)
	}

	if op.LongTimeout {
		options = append(options, grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 20 * time.Minute,
			Time:              10 * time.Second,
			Timeout:           15 * time.Minute,
		}))
	} else {
		options = append(options, grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 10 * time.Minute,
			Time:              10 * time.Second,
			Timeout:           5 * time.Minute,
		}))
	}

	if op.LargeFile {
		options = append(options,
			grpc.InitialWindowSize(32<<20),      // 32 MB
			grpc.InitialConnWindowSize(128<<20), // 128 MB
			grpc.ReadBufferSize(1<<20),          // 1 MB
			grpc.WriteBufferSize(1<<20),         // 1 MB
		)
	}

	s := grpc.NewServer(options...)

	if !op.DisableReflection {
		reflection.Register(s)
	}

	healthService := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s, healthService)

	return s
}
