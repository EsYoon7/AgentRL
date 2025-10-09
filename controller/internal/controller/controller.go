package controller

import (
	"net/http"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/utils"
	"google.golang.org/grpc"
)

type Controller struct {
	Logger             echo.Logger
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

func Setup(e *echo.Echo, grpc *grpc.Server, longTimeout bool) {
	controller := &Controller{
		Logger:             e.Logger,
		Transport:          utils.NewTransport(),
		SessionExpireTime:  5 * time.Minute,
		SessionRemoveTime:  10 * time.Minute,
		CleanInterval:      20 * time.Second,
		HeartbeatTimeout:   11 * time.Second,
		WorkerRemoveTime:   5 * time.Minute,
		SyncInterval:       1 * time.Minute,
		InteractionTimeout: 4 * time.Minute,
	}

	if longTimeout {
		controller.SessionExpireTime = 11 * time.Minute
		controller.SessionRemoveTime = 20 * time.Minute
		controller.InteractionTimeout = 10 * time.Minute
	}

	controller.TaskManager = controller.NewTaskManager()
	controller.SessionManager = controller.NewSessionsManager()

	e.GET("/api/list_workers", controller.handleListWorkers)
	e.GET("/api/list_sessions", controller.handleListSessions)
	e.GET("/api/get_indices", controller.handleGetIndices)
	e.POST("/api/mark_stale", controller.handleMarkStale)
	e.POST("/api/start_sample", controller.handleStartSample)
	e.POST("/api/interact", controller.handleInteract)
	e.POST("/api/cancel", controller.handleCancel)
	e.POST("/api/cancel_all", controller.handleCancelAll)
	e.POST("/api/cancel_notice", controller.handleCancelNotice)
	e.POST("/api/receive_heartbeat", controller.handleReceiveHeartbeat)
	e.POST("/api/clean_worker", controller.handleCleanWorker)
	e.POST("/api/clean_session", controller.handleCleanSession)
	e.POST("/api/sync_all", controller.handleSyncAll)
	e.GET("/api/version", controller.handleGetVersion)

	pb.RegisterControllerServer(grpc, NewGrpcServer(controller))

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
