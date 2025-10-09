package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/sasha-s/go-deadlock"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/types"
	"github.com/thudm/agentrl/controller/internal/utils"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type WorkerStatus int

const (
	WorkerStatusAlive WorkerStatus = iota
	WorkerStatusComa
	WorkerStatusDead
)

type Worker struct {
	controller *Controller
	stream     *grpc.BidiStreamingServer[pb.WorkerStreamEnvelope, pb.WorkerStreamEnvelope]
	pendingRpc map[string]*grpcRequest
	Id         int
	Name       string
	Address    string
	Capacity   int
	Current    int
	LastVisit  time.Time
	Status     WorkerStatus
	Stale      bool
	Lock       deadlock.RWMutex
}

func (w *Worker) Call(ctx context.Context, method string, api string, body interface{}) (io.Reader, error) {
	if strings.HasPrefix(w.Address, "grpc://") {
		// this worker uses gRPC transport
		return w.CallGrpc(ctx, method, api, body)
	}

	reqUrl, err := url.JoinPath(w.Address, api)
	if err != nil {
		return nil, err
	}

	var bodyReader io.Reader
	if body != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(bodyBytes)
	} else {
		bodyReader = http.NoBody
	}

	req, err := http.NewRequestWithContext(ctx, method, reqUrl, bodyReader)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	w.controller.Logger.Debugf("worker %s#%d call %s %s", w.Name, w.Id, method, reqUrl)

	response, err := w.controller.Transport.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	defer response.Body.Close()

	if !(response.StatusCode >= 200 && response.StatusCode < 300) {
		w.controller.Logger.Errorf("worker %s#%d call %s %s failed with status code %d", w.Name, w.Id, method, api, response.StatusCode)
		return nil, echo.NewHTTPError(response.StatusCode)
	}

	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return bytes.NewReader(respBody), nil
}

func (w *Worker) CallGrpc(ctx context.Context, method string, api string, body interface{}) (io.Reader, error) {
	if w.stream == nil {
		return nil, fmt.Errorf("worker %s#%d is not connected to a gRPC stream", w.Name, w.Id)
	}

	requestUuid, err := uuid.NewRandom()
	if err != nil {
		return nil, fmt.Errorf("failed to generate ID for gRPC request: %w", err)
	}
	requestId := requestUuid.String()
	requestType := pb.WorkerStreamEnvelope_REQUEST
	envelope := pb.WorkerStreamEnvelope{
		Id:        &requestId,
		Type:      &requestType,
		Timestamp: timestamppb.Now(),
	}

	// encode json body
	var bodyBytes []byte
	if body != nil {
		bodyBytes, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
	}
	message := &pb.WorkerStreamEnvelope_WorkerRequest{
		Method:   &method,
		Endpoint: &api,
		Json:     bodyBytes,
	}
	envelope.Body = &pb.WorkerStreamEnvelope_WorkerRequest_{WorkerRequest: message}

	// create channel to receive response
	respCh := make(chan *pb.WorkerStreamEnvelope, 1)
	w.Lock.Lock()
	w.pendingRpc[requestId] = &grpcRequest{
		respCh: respCh,
	}
	w.Lock.Unlock()

	// ensure cleanup
	defer func() {
		w.Lock.Lock()
		delete(w.pendingRpc, requestId)
		w.Lock.Unlock()
		close(respCh)
	}()

	err = (*w.stream).Send(&envelope)
	if err != nil {
		return nil, fmt.Errorf("failed to send gRPC request to worker %s#%d: %w", w.Name, w.Id, err)
	}

	select {
	case resp := <-respCh:
		if resp == nil || resp.GetId() != requestId || resp.GetType() != pb.WorkerStreamEnvelope_RESPONSE {
			return nil, fmt.Errorf("invalid gRPC response for request %s", requestId)
		}

		responseMessage := resp.GetWorkerResponse()
		if responseMessage == nil {
			return nil, fmt.Errorf("invalid gRPC response for request %s", requestId)
		}

		responseCode := responseMessage.GetCode()
		if !(responseCode >= 200 && responseCode < 300) {
			w.controller.Logger.Errorf("worker %s#%d call %s %s failed with status code %d", w.Name, w.Id, method, api, responseCode)
			return nil, echo.NewHTTPError(int(responseCode))
		}

		return bytes.NewReader(responseMessage.GetJson()), nil

	case <-ctx.Done():
		return nil, ctx.Err()

	case <-time.After(10 * time.Minute):
		// final timeout to avoid hanging forever
		return nil, fmt.Errorf("gRPC request %s to worker %s#%d closed after 10 minutes", requestId, w.Name, w.Id)
	}
}

func (w *Worker) FinalizeGrpcCall(requestId string, response *pb.WorkerStreamEnvelope) {
	w.Lock.Lock()
	defer w.Lock.Unlock()

	if req, exists := w.pendingRpc[requestId]; exists {
		req.respCh <- response
		delete(w.pendingRpc, requestId)
	} else {
		w.controller.Logger.Warnf("received gRPC response for unknown request %s from worker %s#%d", requestId, w.Name, w.Id)
	}
}

// Sync synchronously calls the worker to update its status.
// Should be called as a goroutine.
func (w *Worker) Sync() {
	if w.Status == WorkerStatusDead {
		// worker has been deleted, no need to sync
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	syncTime := time.Now()
	body, err := w.Call(ctx, http.MethodGet, "get_sessions", nil)

	if w.Status == WorkerStatusDead {
		// worker has been deleted, no need to sync
		return
	}

	w.Lock.Lock()
	defer w.Lock.Unlock()

	if err != nil {
		w.controller.Logger.Errorf("worker %s#%d sync failed: %v", w.Name, w.Id, err)
		w.Status = WorkerStatusComa // mark worker as coma
		return
	}

	var workerSessions map[int]types.TaskIndex
	err = json.NewDecoder(body).Decode(&workerSessions)
	if err != nil {
		w.controller.Logger.Errorf("worker %s#%d sync failed: %v", w.Name, w.Id, err)
		w.Status = WorkerStatusComa // mark worker as coma
		return
	}

	// update worker status
	w.LastVisit = time.Now()
	if w.Status != WorkerStatusAlive {
		w.controller.Logger.Infof("worker %s#%d is alive", w.Name, w.Id)
		w.Status = WorkerStatusAlive
	}

	// sync sessions:
	// - for sessions exist in the worker but not in the controller, add them to the controller
	// - for sessions exist in the controller but not in the worker, remove them from the controller
	// - if there are mismatched sessions, it should not happen, and we cancel all sessions belonging to the worker

	controllerSessions := w.controller.SessionManager.GatherSessions(func(session *Session) bool {
		return session.Worker == w
	}, false)

	for id, session := range controllerSessions {
		_, exists := workerSessions[id]
		if !exists && session.LastUpdate.Before(syncTime) {
			w.controller.SessionManager.FinishSession(session, false, false, false)
		}
	}

	for id, index := range workerSessions {
		w.controller.SessionManager.ResumeSession(w, id, index, syncTime)
	}
}

type WorkerCancelRequest struct {
	SessionId int `json:"session_id"`
}

// Cancel calls the worker to request it cancel a session.
// Should be called as a goroutine.
func (w *Worker) Cancel(sessionId int) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// call cancel and ignore response
	_, err := w.Call(ctx, http.MethodPost, "cancel", WorkerCancelRequest{
		SessionId: sessionId,
	})

	w.Lock.Lock()
	defer w.Lock.Unlock()

	if err != nil {
		w.controller.Logger.Errorf("worker %s#%d cancel session %d failed: %v", w.Name, w.Id, sessionId, err)
		w.Status = WorkerStatusComa // mark worker as coma
		return
	}
}

// CancelWithNotice calls the worker to request it to cancel a session.
// Instead of waiting for its response, the worker will send a notice to the controller when the cancellation is done.
// Should be called as a goroutine.
func (w *Worker) CancelWithNotice(sessionId int) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// call cancel_with_notice and ignore response
	// to ensure proper fallback,
	// this method should only success when the session exists in the worker
	// and the worker started to handle its cancellation
	_, err := w.Call(ctx, http.MethodPost, "cancel_with_notice", WorkerCancelRequest{
		SessionId: sessionId,
	})

	if err != nil {
		// if cancel_with_notice is not available,
		// fallback to ordinary cancel
		w.Cancel(sessionId)

		// then finish the session (if not already)
		session, exists := w.controller.SessionManager.GetSession(sessionId)
		if exists {
			w.controller.SessionManager.FinishSession(session, false, false, true)
		}
	}
}

// CancelAll calls the worker to request it cancel all sessions.
// Should be called as a goroutine.
func (w *Worker) CancelAll(critical bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// call cancel_all and ignore response
	_, err := w.Call(ctx, http.MethodPost, "cancel_all", nil)

	w.Lock.Lock()
	defer w.Lock.Unlock()

	if err != nil {
		w.controller.Logger.Errorf("worker %s#%d cancel_all failed: %v", w.Name, w.Id, err)
		if critical {
			// this argument marks that the worker's response here is critical
			// if error occurs, the worker is considered dead
			go w.controller.TaskManager.RemoveWorker(w)
		} else {
			// mark worker as coma
			w.Status = WorkerStatusComa
		}
		return
	}
}

// CancelAllWithNotice calls the worker to request it cancel all sessions.
// Instead of waiting for its response, the worker will send a notice to the controller when the cancellation is done.
// Should be called as a goroutine.
func (w *Worker) CancelAllWithNotice(critical bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// call cancel_all_with_notice and ignore response
	// to ensure proper fallback,
	// this method should only success when the session exists in the worker
	// and the worker started to handle its cancellation
	_, err := w.Call(ctx, http.MethodPost, "cancel_all_with_notice", nil)

	if err != nil {
		// if cancel_all_with_notice is not available,
		// fallback to ordinary cancel_all
		w.CancelAll(critical)

		// then finish the sessions (if not already)
		w.controller.SessionManager.FinishSessions(func(session *Session) bool {
			return session.Worker == w
		}, false)
	}
}

// SetStatus locks the worker and sets its status.
func (w *Worker) SetStatus(status WorkerStatus) {
	w.Lock.Lock()
	defer w.Lock.Unlock()

	if w.Status == status || w.Status == WorkerStatusDead {
		return // no change
	}

	w.Status = status
	w.LastVisit = time.Now()
}

func (w *Worker) Dump() types.WorkerData {
	var status string
	switch w.Status {
	case WorkerStatusAlive:
		status = "ALIVE"
	case WorkerStatusComa:
		status = "COMA"
	case WorkerStatusDead:
		status = "DEAD"
	}

	return types.WorkerData{
		Id:        w.Id,
		Address:   w.Address,
		Capacity:  w.Capacity,
		Current:   w.Current,
		LastVisit: utils.FormatTime(w.LastVisit),
		Status:    status,
		Stale:     w.Stale,
	}
}
