package snapshots

import (
	"github.com/labstack/echo/v4"
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
	Server       *Server
}

type ManagerOptions struct {
	Logger             *zap.SugaredLogger
	NodeRegistry       *cluster.NodeRegistry
	HttpServer         *echo.Echo
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

	server := &Server{
		logger:  op.Logger,
		manager: manager,
	}
	snapshots_v1.RegisterSnapshotsManagerServer(op.GrpcServer, server)
	manager.Server = server

	handler := &httpHandler{
		server: server,
	}
	handler.RegisterHttpRoutes(op.HttpServer)

	return manager
}

func (m *Manager) Close() {
	if m.Database != nil {
		m.Database.Close()
	}
}
