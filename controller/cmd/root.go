package cmd

import (
	"log"

	"github.com/labstack/echo/v4"
	"github.com/spf13/cobra"
	"github.com/thudm/agentrl/controller/internal/utils"
	"google.golang.org/grpc"
)

type GlobalFlags struct {
	Host         string
	Port         uint64
	Debug        bool
	Deadlock     bool
	DashboardDev bool
}

var flags = GlobalFlags{}

var echoServer *echo.Echo

var grpcServer *grpc.Server

var rootCmd = &cobra.Command{
	Use: "agentrl [flags] [command]",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		utils.ConfigureDeadlock(flags.Deadlock)
		echoServer = createEchoServer(&flags)
		grpcServer = createGrpcServer(&flags)
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&flags.Host, "host", "", "Host to bind to")
	rootCmd.PersistentFlags().Uint64Var(&flags.Port, "port", 5020, "Port to bind to")
	rootCmd.PersistentFlags().BoolVar(&flags.Debug, "debug", false, "Enable debug logging")
	rootCmd.PersistentFlags().BoolVar(&flags.Deadlock, "deadlock", false, "Enable deadlock detection")
	rootCmd.PersistentFlags().BoolVar(&flags.DashboardDev, "dashboard-dev", false, "Enable dashboard dev mode")
}
