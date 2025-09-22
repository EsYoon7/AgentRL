package server

import (
	"net/http"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/thudm/agentrl/controller/internal/middlewares"
)

type HttpOptions struct {
	Dashboard    bool
	DashboardDev bool
	Debug        bool
	LongTimeout  bool
}

func NewHttpServer(op HttpOptions) (*echo.Echo, *http.Server) {
	e := echo.New()
	e.Use(middleware.Recover())
	e.Use(middleware.Gzip())

	if op.Debug {
		e.Debug = true
	} else {
		e.Debug = false
	}

	if op.Dashboard {
		e.Use(middlewares.Dashboard(op.DashboardDev))
	}

	server := &http.Server{
		Handler:           e,
		IdleTimeout:       10 * time.Minute,
		ReadTimeout:       5 * time.Minute,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      5 * time.Minute,
	}

	if op.LongTimeout {
		server.IdleTimeout = 20 * time.Minute
		server.ReadTimeout = 10 * time.Minute
		server.WriteTimeout = 10 * time.Minute
	}

	return e, server
}
