package cmd

import (
	"github.com/spf13/cobra"
	"github.com/thudm/agentrl/controller/internal/cluster"
	"github.com/thudm/agentrl/controller/internal/server"
	"github.com/thudm/agentrl/controller/internal/snapshots"
)

type SnapshotsFlags struct {
	DatabaseConnection string
}

var snapshotsFlags = SnapshotsFlags{}

var snapshotsCmd = &cobra.Command{
	Use:   "snapshots [flags] <store_dir>",
	Short: "Snapshots manager",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		grpc := server.NewGrpcServer(server.GrpcOptions{
			Logger:    logger.Named("grpc"),
			LargeFile: true,
		})

		registry := cluster.NewMemberlist(cluster.MemberlistOptions{
			BindHost: flags.Host,
			BindPort: flags.GossipPort,
			Join:     flags.MemberlistJoin,
			Logger:   logger.Named("memberlist"),
			NodeInfo: &cluster.NodeInfo{
				ServicePort: flags.Port,
			},
		})

		manager := snapshots.NewManager(snapshots.ManagerOptions{
			Logger:             logger.Named("snapshots"),
			NodeRegistry:       registry,
			GrpcServer:         grpc,
			DatabaseConnection: snapshotsFlags.DatabaseConnection,
			StoreDirectory:     args[0],
		})
		defer manager.Close()

		server.StartServer(server.StartServerOptions{
			Host:       flags.Host,
			Port:       flags.Port,
			GrpcServer: grpc,
			Logger:     logger.Named("server"),
		})
	},
}

func init() {
	rootCmd.AddCommand(snapshotsCmd)
	snapshotsCmd.PersistentFlags().StringVar(&snapshotsFlags.DatabaseConnection, "database", "", "postgresql connection string")
	snapshotsCmd.MarkPersistentFlagRequired("database")
}
