package server

import (
	"errors"
	"net"
	"net/http"
	"strconv"

	"github.com/soheilhy/cmux"
	"github.com/thudm/agentrl/controller/internal/utils"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

type StartServerOptions struct {
	Host       string
	Port       uint16
	HttpServer *http.Server
	GrpcServer *grpc.Server
	Logger     *zap.SugaredLogger
}

func StartServer(op *StartServerOptions) {
	if op.HttpServer == nil && op.GrpcServer == nil {
		op.Logger.Fatal("no servers to start")
	}

	addr := net.JoinHostPort(op.Host, strconv.Itoa(int(op.Port)))
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		op.Logger.Fatalf("failed to listen on %s:%d: %v", op.Host, op.Port, err)
	}

	listener := utils.KeepAliveListener{
		TCPListener: ln.(*net.TCPListener),
	}

	m := cmux.New(listener)

	if op.GrpcServer != nil {
		grpcL := m.MatchWithWriters(cmux.HTTP2MatchHeaderFieldSendSettings("content-type", "application/grpc"))

		go func() {
			if err = op.GrpcServer.Serve(grpcL); !errors.Is(err, grpc.ErrServerStopped) {
				op.Logger.Fatalf("failed to serve gRPC: %v", err)
			}
		}()

		op.Logger.Infof("gRPC server started at %s:%d", op.Host, op.Port)
	}

	if op.HttpServer != nil {
		httpL := m.Match(cmux.Any())

		go func() {
			if err = op.HttpServer.Serve(httpL); !errors.Is(err, http.ErrServerClosed) {
				op.Logger.Fatalf("failed to serve HTTP: %v", err)
			}
		}()

		op.Logger.Infof("HTTP server started at %s:%d", op.Host, op.Port)
	}

	if err = m.Serve(); !errors.Is(err, cmux.ErrServerClosed) {
		op.Logger.Fatalf("failed to serve: %v", err)
	}
}
