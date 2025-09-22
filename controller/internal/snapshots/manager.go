package snapshots

import (
	"github.com/thudm/agentrl/controller/internal/cluster"
	"github.com/thudm/agentrl/controller/internal/pb/snapshots_v1"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

type Manager struct {
	Logger       *zap.SugaredLogger
	NodeRegistry *cluster.NodeRegistry
	Database     *Database
	Store        *Store
}

type ManagerOptions struct {
	Logger             *zap.SugaredLogger
	NodeRegistry       *cluster.NodeRegistry
	GrpcServer         *grpc.Server
	DatabaseConnection string
	StoreDirectory     string
}

func NewManager(op ManagerOptions) *Manager {
	manager := &Manager{
		Logger:       op.Logger,
		NodeRegistry: op.NodeRegistry,
		Database:     NewDatabase(op.Logger, op.DatabaseConnection),
		Store:        NewStore(op.Logger, op.StoreDirectory),
	}

	snapshots_v1.RegisterSnapshotsManagerServer(op.GrpcServer, &Server{
		logger:  op.Logger,
		manager: manager,
	})

	return manager
}

func (m *Manager) Close() {
	if m.Database != nil {
		m.Database.Close()
	}
}
