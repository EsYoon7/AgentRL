package cmd

import (
	"errors"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/labstack/gommon/log"
	"github.com/soheilhy/cmux"
	"github.com/thudm/agentrl/controller/internal/middlewares"
	"github.com/thudm/agentrl/controller/internal/utils"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
)

func createEchoServer(globalFlags *GlobalFlags) *echo.Echo {
	e := echo.New()
	e.Use(middleware.Recover())
	e.Use(middleware.Gzip())

	if globalFlags.Debug {
		e.Logger.SetLevel(log.DEBUG)
		e.Debug = true
	} else {
		e.Logger.SetLevel(log.INFO)
		e.Debug = false
	}

	e.Use(middlewares.Dashboard(globalFlags.DashboardDev))

	return e
}

func createGrpcServer(_ *GlobalFlags) *grpc.Server {
	keepAliveEnforcementOption := grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second,
		PermitWithoutStream: true,
	})

	keepAliveOption := grpc.KeepaliveParams(keepalive.ServerParameters{
		MaxConnectionIdle: 10 * time.Minute,
		Time:              10 * time.Second,
		Timeout:           5 * time.Minute,
	})

	maxSizeOption := grpc.MaxRecvMsgSize(100 * 1024 * 1024) // 100 MB

	return grpc.NewServer(keepAliveEnforcementOption, keepAliveOption, maxSizeOption)
}

func startServer(flags *GlobalFlags) {
	server := &http.Server{
		Addr:              net.JoinHostPort(flags.Host, strconv.FormatUint(flags.Port, 10)),
		Handler:           echoServer,
		IdleTimeout:       10 * time.Minute,
		ReadTimeout:       5 * time.Minute,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      5 * time.Minute,
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(flags.Host, strconv.FormatUint(flags.Port, 10)))
	if err != nil {
		echoServer.Logger.Fatalf("Failed to listen on %s:%d: %v", flags.Host, flags.Port, err)
	}

	listener := utils.KeepAliveListener{
		TCPListener: ln.(*net.TCPListener),
	}

	m := cmux.New(listener)
	grpcL := m.MatchWithWriters(cmux.HTTP2MatchHeaderFieldSendSettings("content-type", "application/grpc"))
	httpL := m.Match(cmux.Any())

	go func() {
		if err = server.Serve(httpL); !errors.Is(err, http.ErrServerClosed) {
			echoServer.Logger.Fatalf("Failed to serve HTTP: %v", err)
		}
	}()
	go func() {
		if err = grpcServer.Serve(grpcL); !errors.Is(err, grpc.ErrServerStopped) {
			echoServer.Logger.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()

	echoServer.Logger.Infof("HTTP/gRPC server started at %s:%d", flags.Host, flags.Port)
	if err = m.Serve(); !errors.Is(err, cmux.ErrServerClosed) {
		echoServer.Logger.Fatalf("Failed to serve: %v", err)
	}
}
