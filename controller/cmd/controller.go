package cmd

import (
	"github.com/spf13/cobra"
	"github.com/thudm/agentrl/controller/internal/controller"
)

type ControllerFlags struct {
	LongTimeout bool
}

var controllerFlags = ControllerFlags{}

var controllerCmd = &cobra.Command{
	Use:   "controller",
	Short: "AgentRL controller",
	Args:  cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		controller.Setup(echoServer, grpcServer, controllerFlags.LongTimeout)
		startServer(&flags)
	},
}

func init() {
	rootCmd.AddCommand(controllerCmd)
	controllerCmd.PersistentFlags().BoolVar(&controllerFlags.LongTimeout, "long-timeout", false, "Enable long timeout for interactions")
}
