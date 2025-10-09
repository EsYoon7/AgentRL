package middlewares

import (
	"log"
	"net/url"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/thudm/agentrl/controller/dashboard"
)

var viteDevServer = "http://localhost:5173"

func Dashboard(isDev bool) func(echo.HandlerFunc) echo.HandlerFunc {
	skipper := func(c echo.Context) bool {
		return strings.HasPrefix(c.Path(), "/api/")
	}

	if isDev {
		serverUrl, err := url.Parse(viteDevServer)
		if err != nil {
			log.Fatalf("Invalid Vite dev server URL: %v", err)
		}

		balancer := middleware.NewRoundRobinBalancer([]*middleware.ProxyTarget{
			{
				URL: serverUrl,
			},
		})

		return middleware.ProxyWithConfig(middleware.ProxyConfig{
			Skipper:  skipper,
			Balancer: balancer,
		})
	}

	return middleware.StaticWithConfig(middleware.StaticConfig{
		Skipper:    skipper,
		Filesystem: dashboard.FS(),
	})
}
