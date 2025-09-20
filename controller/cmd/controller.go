package cmd

import (
	"github.com/spf13/cobra"
	"github.com/thudm/agentrl/controller/internal/controller"
	"github.com/thudm/agentrl/controller/internal/server"
)

type ControllerFlags struct {
	Dashboard    bool
	DashboardDev bool
	LongTimeout  bool
}

var controllerFlags = ControllerFlags{}

var controllerCmd = &cobra.Command{
	Use:   "controller [flags]",
	Short: "AgentRL controller",
	Args:  cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		echo, http := server.CreateHttpServer(&server.HttpOptions{
			Dashboard:    controllerFlags.Dashboard,
			DashboardDev: controllerFlags.DashboardDev,
			Debug:        flags.Debug,
			LongTimeout:  controllerFlags.LongTimeout,
		})

		grpc := server.CreateGrpcServer()

		controller.Setup(&controller.Options{
			Logger:      logger.Named("controller"),
			EchoServer:  echo,
			GrpcServer:  grpc,
			LongTimeout: controllerFlags.LongTimeout,
		})

		server.StartServer(&server.StartServerOptions{
			Host:       flags.Host,
			Port:       flags.Port,
			HttpServer: http,
			GrpcServer: grpc,
			Logger:     logger.Named("server"),
		})
	},
}

func init() {
	rootCmd.AddCommand(controllerCmd)
	controllerCmd.PersistentFlags().BoolVar(&controllerFlags.Dashboard, "dashboard", true, "enable dashboard")
	controllerCmd.PersistentFlags().BoolVar(&controllerFlags.DashboardDev, "dashboard-dev", false, "enable dashboard dev mode")
	controllerCmd.PersistentFlags().BoolVar(&controllerFlags.LongTimeout, "long-timeout", false, "enable long timeout for interactions")
}
