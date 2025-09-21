package controller

import (
	"net/http"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/thudm/agentrl/controller/internal/pb/controller_v1"
	"github.com/thudm/agentrl/controller/internal/utils"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

type Controller struct {
	Logger             *zap.SugaredLogger
	Transport          *http.Transport
	TaskManager        *TaskManager
	SessionManager     *SessionManager
	SessionExpireTime  time.Duration
	SessionRemoveTime  time.Duration
	CleanInterval      time.Duration
	HeartbeatTimeout   time.Duration
	WorkerRemoveTime   time.Duration
	SyncInterval       time.Duration
	InteractionTimeout time.Duration
}

type Options struct {
	Logger      *zap.SugaredLogger
	EchoServer  *echo.Echo
	GrpcServer  *grpc.Server
	LongTimeout bool
}

func Setup(op *Options) {
	controller := &Controller{
		Logger:             op.Logger,
		Transport:          utils.NewTransport(),
		SessionExpireTime:  5 * time.Minute,
		SessionRemoveTime:  10 * time.Minute,
		CleanInterval:      20 * time.Second,
		HeartbeatTimeout:   11 * time.Second,
		WorkerRemoveTime:   5 * time.Minute,
		SyncInterval:       1 * time.Minute,
		InteractionTimeout: 4 * time.Minute,
	}

	if op.LongTimeout {
		controller.SessionExpireTime = 11 * time.Minute
		controller.SessionRemoveTime = 20 * time.Minute
		controller.InteractionTimeout = 10 * time.Minute
	}

	controller.TaskManager = controller.NewTaskManager()
	controller.SessionManager = controller.NewSessionsManager()

	op.EchoServer.GET("/api/list_workers", controller.handleListWorkers)
	op.EchoServer.GET("/api/list_sessions", controller.handleListSessions)
	op.EchoServer.GET("/api/get_indices", controller.handleGetIndices)
	op.EchoServer.POST("/api/mark_stale", controller.handleMarkStale)
	op.EchoServer.POST("/api/start_sample", controller.handleStartSample)
	op.EchoServer.POST("/api/interact", controller.handleInteract)
	op.EchoServer.POST("/api/cancel", controller.handleCancel)
	op.EchoServer.POST("/api/cancel_all", controller.handleCancelAll)
	op.EchoServer.POST("/api/cancel_notice", controller.handleCancelNotice)
	op.EchoServer.POST("/api/receive_heartbeat", controller.handleReceiveHeartbeat)
	op.EchoServer.POST("/api/clean_worker", controller.handleCleanWorker)
	op.EchoServer.POST("/api/clean_session", controller.handleCleanSession)
	op.EchoServer.POST("/api/sync_all", controller.handleSyncAll)
	op.EchoServer.GET("/api/version", controller.handleGetVersion)

	controller_v1.RegisterControllerServer(op.GrpcServer, &GrpcServer{
		controller: controller,
	})

	// background tasks
	go func() {
		for {
			time.Sleep(controller.CleanInterval)
			controller.SessionManager.CleanSessions()
		}
	}()

	go func() {
		for {
			time.Sleep(controller.CleanInterval)
			controller.TaskManager.CleanWorkers()
		}
	}()

	go func() {
		for {
			time.Sleep(controller.SyncInterval)
			controller.TaskManager.CallSyncAll()
		}
	}()
}
