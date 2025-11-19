package controller

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/labstack/echo/v4"
	"github.com/thudm/agentrl/controller/internal/types"
)

func (controller *Controller) handleListWorkers(c echo.Context) error {
	return c.JSON(http.StatusOK, controller.TaskManager.Dump())
}

func (controller *Controller) handleListSessions(c echo.Context) error {
	return c.JSON(http.StatusOK, controller.SessionManager.Dump())
}

func (controller *Controller) handleGetIndices(c echo.Context) error {
	name := c.QueryParam("name")

	indices, err := controller.TaskManager.DumpIndices(name)
	if err != nil {
		controller.Logger.Errorf("failed to get indices: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}

	return c.JSON(http.StatusOK, indices)
}

type MarkStaleRequest struct {
	Name     string `json:"name"`
	WorkerId int    `json:"worker_id"`
	Stale    bool   `json:"stale"`
}

func (controller *Controller) handleMarkStale(c echo.Context) error {
	var request MarkStaleRequest
	if err := c.Bind(&request); err != nil {
		controller.Logger.Errorf("failed to bind request: %v", err)
		return &echo.HTTPError{
			Internal: err,
			Message:  "invalid request",
			Code:     http.StatusBadRequest,
		}
	}

	err := controller.TaskManager.MarkWorkerStale(request.Name, request.WorkerId, request.Stale)

	if err != nil {
		controller.Logger.Errorf("failed to mark worker as stale: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}

	return c.JSON(http.StatusOK, nil)
}

type StartSampleRequest struct {
	Name       string          `json:"name"`
	Index      types.TaskIndex `json:"index"`
	CustomTask interface{}     `json:"custom_task"`
}

type StartSampleResponse struct {
	// only preserve these fields
	Messages []types.ChatMessage `json:"messages"`
	Tools    interface{}         `json:"tools"`
}

func (controller *Controller) handleStartSample(c echo.Context) error {
	var request StartSampleRequest
	if err := c.Bind(&request); err != nil {
		controller.Logger.Errorf("failed to bind request: %v", err)
		return &echo.HTTPError{
			Internal: err,
			Message:  "invalid request",
			Code:     http.StatusBadRequest,
		}
	}

	// dispatch task to an idle worker
	session, err := controller.TaskManager.DispatchTask(request.Name, request.Index, request.CustomTask)
	if err != nil {
		controller.Logger.Errorf("failed to dispatch task: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}
	c.Response().Header().Set(SessionIdKey, strconv.Itoa(session.Id))

	result, err := session.Interact(nil)
	if err != nil {
		// failed to start sample, the session should be removed
		controller.SessionManager.FinishSession(session, false, false, true)

		// if error status is 406, the worker is full, we need to sync its status
		var httpError *echo.HTTPError
		if session.Worker != nil && errors.As(err, &httpError) && httpError.Code == http.StatusNotAcceptable {
			controller.Logger.Infof("Worker %s#%d is unexpectedly full, syncing...", session.Worker.Name, session.Worker.Id)
			go session.Worker.Sync()
		}

		controller.Logger.Errorf("Session %d failed to start sample: %v", session.Id, err)
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to start sample")
	}

	return c.JSON(http.StatusOK, StartSampleResponse{
		Messages: result.Messages,
		Tools:    result.Tools,
	})
}

type InteractRequest struct {
	Messages []types.ChatMessage `json:"messages"`
}

type InteractResponse struct {
	Finish   bool                   `json:"finish"`
	Reward   float64                `json:"reward"`
	Status   SampleStatus           `json:"status"`
	Messages []types.ChatMessage    `json:"messages"`
	Metrics  map[string]interface{} `json:"metrics"`
}

func (controller *Controller) handleInteract(c echo.Context) error {
	var request InteractRequest
	if err := c.Bind(&request); err != nil {
		controller.Logger.Errorf("failed to bind request: %v", err)
		return &echo.HTTPError{
			Internal: err,
			Message:  "invalid request",
			Code:     http.StatusBadRequest,
		}
	}

	sessionId, err := strconv.Atoi(c.Request().Header.Get(SessionIdKey))
	if err != nil {
		controller.Logger.Errorf("failed to parse session id: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, "invalid session id")
	}

	session, exists := controller.SessionManager.GetSession(sessionId)
	if !exists {
		controller.Logger.Errorf("session %d not found", sessionId)
		return echo.NewHTTPError(http.StatusBadRequest, "session not found")
	}

	result, err := session.Interact(request.Messages)

	if err != nil {
		controller.Logger.Errorf("failed to interact with session %d: %v", sessionId, err)
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to interact with session")
	}

	return c.JSON(http.StatusOK, InteractResponse{
		Finish:   result.Finish,
		Reward:   result.Reward,
		Status:   result.Status,
		Messages: result.Messages,
		Metrics:  result.Metric,
	})
}

func (controller *Controller) handleCancel(c echo.Context) error {
	sessionId, err := strconv.Atoi(c.Request().Header.Get(SessionIdKey))
	if err != nil {
		controller.Logger.Errorf("failed to parse session id: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, "invalid session id")
	}

	session, exists := controller.SessionManager.GetSession(sessionId)
	if !exists {
		controller.Logger.Errorf("session %d not found", sessionId)
		return echo.NewHTTPError(http.StatusBadRequest, "session not found")
	}

	controller.SessionManager.FinishSession(session, true, true, true)

	return c.JSON(http.StatusOK, nil)
}

func (controller *Controller) handleCancelAll(c echo.Context) error {
	if len(controller.TaskManager.Tasks) > 1 {
		// multitask controller, should confirm before cancel all

		if taskName := c.QueryParam("name"); taskName != "" {
			controller.TaskManager.CallCancelAllFor(taskName)
		} else if force, _ := strconv.ParseBool(c.QueryParam("force")); force {
			controller.TaskManager.CallCancelAll()
		} else {
			return echo.NewHTTPError(http.StatusForbidden, "multiple tasks are running! confirmation required.")
		}
	} else {
		controller.TaskManager.CallCancelAll()
	}

	return c.JSON(http.StatusOK, nil)
}

func (controller *Controller) handleCancelNotice(c echo.Context) error {
	sessionId, err := strconv.Atoi(c.Request().Header.Get(SessionIdKey))
	if err != nil {
		controller.Logger.Errorf("failed to parse session id: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, "invalid session id")
	}

	err = controller.handleCancelNoticeGeneric(sessionId)
	if err != nil {
		return err
	}
	return c.JSON(http.StatusOK, nil)
}

func (controller *Controller) handleCancelNoticeGeneric(sessionId int) error {
	session, exists := controller.SessionManager.GetSession(sessionId)
	if exists {
		controller.SessionManager.FinishSession(session, false, false, true)
	} else {
		controller.Logger.Warnf("cancel notice received for session %d but the session is not found", sessionId)
	}

	return nil
}

type ReceiveHeartbeatRequest struct {
	Name        string            `json:"name"`
	Address     string            `json:"address"`
	Concurrency int               `json:"concurrency"`
	Indices     []types.TaskIndex `json:"indices"`
}

func (controller *Controller) handleReceiveHeartbeat(c echo.Context) error {
	var request ReceiveHeartbeatRequest
	if err := c.Bind(&request); err != nil {
		controller.Logger.Errorf("failed to bind request: %v", err)
		return &echo.HTTPError{
			Internal: err,
			Message:  "invalid request",
			Code:     http.StatusBadRequest,
		}
	}

	_, err := controller.TaskManager.CreateOrValidateTask(request.Name, request.Indices)
	if err != nil {
		controller.Logger.Errorf("failed to create or validate task: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}

	_, err = controller.TaskManager.UpdateWorker(request.Name, request.Address, request.Concurrency, nil)
	if err != nil {
		controller.Logger.Errorf("failed to update worker: %v", err)
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}

	return c.JSON(http.StatusOK, nil)
}

func (controller *Controller) handleCleanWorker(c echo.Context) error {
	controller.TaskManager.CleanWorkers()

	return c.JSON(http.StatusOK, nil)
}

func (controller *Controller) handleCleanSession(c echo.Context) error {
	controller.SessionManager.CleanSessions()

	return c.JSON(http.StatusOK, nil)
}

func (controller *Controller) handleSyncAll(c echo.Context) error {
	controller.TaskManager.CallSyncAll()

	return c.JSON(http.StatusOK, nil)
}

type GetVersionResponse struct {
	Variant string `json:"variant"`
	Version int    `json:"version"`
}

// handleGetVersion this api is added in agentrl and does not exist in agentbench.
// The purpose is to enable the dashboard to distinguish the type of its backend.
func (controller *Controller) handleGetVersion(c echo.Context) error {
	return c.JSON(http.StatusOK, GetVersionResponse{
		Variant: "nova",
		Version: 1,
	})
}
