package cmd

import (
	"log"

	"github.com/sasha-s/go-deadlock"
	"github.com/spf13/cobra"
	"github.com/thudm/agentrl/controller/internal/utils"
	"go.uber.org/zap"
)

type GlobalFlags struct {
	Host            string
	Port            uint16
	Debug           bool
	Deadlock        bool
	GossipAdvertise string
	GossipPort      uint16
	MemberlistJoin  []string
}

var flags = GlobalFlags{}

var logger *zap.SugaredLogger

var rootCmd = &cobra.Command{
	Use: "agentrl [globalFlags] [command]",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		logger = utils.NewLogger(flags.Debug)
		deadlock.Opts.Disable = !flags.Deadlock
	},
	PersistentPostRun: func(cmd *cobra.Command, args []string) {
		_ = logger.Sync() // flush logger when the program exits
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&flags.Host, "host", "", "host to bind to")
	rootCmd.PersistentFlags().Uint16Var(&flags.Port, "port", 5020, "port to bind to")
	rootCmd.PersistentFlags().BoolVar(&flags.Debug, "debug", false, "enable debug logging")
	rootCmd.PersistentFlags().BoolVar(&flags.Deadlock, "deadlock", false, "enable deadlock detection")
	rootCmd.PersistentFlags().StringVar(&flags.GossipAdvertise, "gossip-advertise", "", "gossip advertise address")
	rootCmd.PersistentFlags().Uint16Var(&flags.GossipPort, "gossip-port", 0, "gossip bind port")
	rootCmd.PersistentFlags().StringSliceVar(&flags.MemberlistJoin, "join", []string{}, "servers to join")
}
