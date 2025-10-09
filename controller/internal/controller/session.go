package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/sasha-s/go-deadlock"
	"github.com/thudm/agentrl/controller/internal/types"
	"github.com/thudm/agentrl/controller/internal/utils"
)

const SessionIdKey = "session_id"

type SampleStatus string

const (
	SampleStatusRunning               SampleStatus = "running"
	SampleStatusCompleted                          = "completed"
	SampleStatusAgentContextLimit                  = "agent context limit"
	SampleStatusAgentValidationFailed              = "agent validation failed"
	SampleStatusAgentInvalidAction                 = "agent invalid action"
	SampleStatusTaskLimitReached                   = "task limit reached"
	SampleStatusUnknown                            = "unknown"
	SampleStatusTaskError                          = "task error"
	SampleStatusCancelled                          = "cancelled"
)

type Session struct {
	controller  *Controller
	Id          int
	Worker      *Worker
	Index       types.TaskIndex
	CustomTask  interface{}
	Start       time.Time
	LastUpdate  time.Time
	Interacting bool
	Cancelling  bool
	Finished    bool
	Lock        deadlock.RWMutex
	Context     context.Context
	CancelFunc  context.CancelFunc
}

type WorkerStartSampleRequest struct {
	Index      types.TaskIndex `json:"index"`
	CustomTask interface{}     `json:"custom_task,omitempty"`
	SessionId  int             `json:"session_id"`
}

type WorkerInteractRequest struct {
	Messages  []types.ChatMessage `json:"messages"`
	SessionId int                 `json:"session_id"`
}

type WorkerEnvResponse struct {
	Messages []types.ChatMessage    `json:"messages"`
	Tools    interface{}            `json:"tools"`
	Status   SampleStatus           `json:"status"`
	Error    string                 `json:"error,omitempty"`
	Finish   bool                   `json:"finish,omitempty"`
	Reward   float64                `json:"reward,omitempty"`
	Metric   map[string]interface{} `json:"metric,omitempty"`
}

type WorkerInteractResponse struct {
	EnvOut WorkerEnvResponse `json:"env_out"`
}

func (s *Session) Interact(messages []types.ChatMessage) (*WorkerEnvResponse, error) {
	s.Lock.Lock()

	if s.Interacting {
		// simultaneous interaction is not allowed
		s.Lock.Unlock()
		return nil, fmt.Errorf("session %d is already interacting", s.Id)
	}

	if s.Cancelling {
		// session is being cancelled, cannot interact
		s.Lock.Unlock()
		return nil, fmt.Errorf("session %d is being cancelled", s.Id)
	}

	s.LastUpdate = time.Now()
	s.Interacting = true
	s.Lock.Unlock()

	defer func() {
		s.Lock.Lock()
		defer s.Lock.Unlock()

		duration := time.Since(s.LastUpdate)
		s.controller.Logger.Debugf("Session %d finished interaction in %.2fs", s.Id, duration.Seconds())

		s.LastUpdate = time.Now()
		s.Interacting = false
	}()

	ctx, cancel := context.WithTimeout(s.Context, s.controller.InteractionTimeout)
	defer cancel()

	var response WorkerEnvResponse
	if messages == nil {
		body, err := s.Worker.Call(ctx, http.MethodPost, "start_sample", WorkerStartSampleRequest{
			Index:      s.Index,
			CustomTask: s.CustomTask,
			SessionId:  s.Id,
		})

		if err != nil {
			return nil, err
		}

		if err = json.NewDecoder(body).Decode(&response); err != nil {
			return nil, err
		}
	} else {
		body, err := s.Worker.Call(ctx, http.MethodPost, "interact", WorkerInteractRequest{
			Messages:  messages,
			SessionId: s.Id,
		})

		if err != nil {
			return nil, err
		}

		var rawResponse WorkerInteractResponse
		if err = json.NewDecoder(body).Decode(&rawResponse); err != nil {
			return nil, err
		}

		response = rawResponse.EnvOut
	}

	if response.Status != SampleStatusRunning {
		// session finished, should be removed
		s.controller.Logger.Infof("Session %d finished with status %s", s.Id, response.Status)
		s.controller.SessionManager.FinishSession(s, false, false, true)
	} else if len(response.Messages) == 0 {
		// no messages returned, indicating worker error
		s.controller.Logger.Errorf("Session %d interaction returned no messages", s.Id)
		return nil, fmt.Errorf("worker returned no messages")
	}

	return &response, nil
}

func (s *Session) Dump() types.SessionData {
	return types.SessionData{
		Name:       s.Worker.Name,
		Index:      s.Index,
		CustomTask: s.CustomTask,
		Start:      utils.FormatTime(s.Start),
		LastUpdate: utils.FormatTime(s.LastUpdate),
		WorkerId:   s.Worker.Id,
		Locked:     s.Interacting || s.Cancelling,
		Cancelling: s.Cancelling,
	}
}

type SessionManager struct {
	controller    *Controller
	nextSessionId atomic.Int32
	Sessions      map[int]*Session
	Lock          deadlock.RWMutex
}

func (sm *SessionManager) CreateSession(worker *Worker, index types.TaskIndex, customTask interface{}) *Session {
	sm.Lock.Lock()
	defer sm.Lock.Unlock()

	sessionId := int(sm.nextSessionId.Add(1))
	session := &Session{
		controller:  sm.controller,
		Id:          sessionId,
		Worker:      worker,
		Index:       index,
		CustomTask:  customTask,
		Interacting: false,
		Cancelling:  false,
		Finished:    false,
		Start:       time.Now(),
		LastUpdate:  time.Now(),
	}
	session.Context, session.CancelFunc = context.WithCancel(context.Background())

	sm.Sessions[sessionId] = session
	sm.controller.Logger.Infof("Created session %d for task %s, index %s, worker %d", session.Id, session.Worker.Name, session.Index, session.Worker.Id)

	return session
}

func (sm *SessionManager) ResumeSession(worker *Worker, sessionId int, index types.TaskIndex, syncTime time.Time) {
	sm.Lock.Lock()
	defer sm.Lock.Unlock()

	session, exists := sm.Sessions[sessionId]
	if exists {
		// session already exists
		session.Lock.Lock()
		defer session.Lock.Unlock()

		if session.Worker != worker || session.Index != index {
			// session mismatch, should not happen
			// do hard sync on worker
			sm.controller.Logger.Errorf("worker %s#%d session %d mismatch, hard sync", worker.Name, worker.Id, sessionId)
			go func() {
				worker.CancelAll(true)
				sm.controller.SessionManager.FinishSessions(func(session *Session) bool {
					return session.Worker == worker
				}, false)
			}()
			return
		}

		if !session.Finished {
			// session is not finished, do nothing
			return
		}

		if session.LastUpdate.After(syncTime) {
			// session is finished and should not be resumed
			return
		}

		// session is actually not finished, try to resume
		session.Context, session.CancelFunc = context.WithCancel(context.Background())
		session.LastUpdate = time.Now()
		session.Finished = false
	} else {
		// session does not exist, create a new one with the same ID
		session = &Session{
			controller:  sm.controller,
			Id:          sessionId,
			Worker:      worker,
			Index:       index,
			CustomTask:  nil,
			Interacting: false,
			Cancelling:  false,
			Finished:    false,
			Start:       time.Now(),
			LastUpdate:  time.Now(),
		}
		session.Context, session.CancelFunc = context.WithCancel(context.Background())
		sm.Sessions[sessionId] = session
	}

	sm.controller.Logger.Infof("Resumed session %d for task %s, index %s, worker %d", session.Id, session.Worker.Name, session.Index, session.Worker.Id)
	worker.Current++
}

func (sm *SessionManager) GetSession(id int) (*Session, bool) {
	sm.Lock.RLock()
	defer sm.Lock.RUnlock()

	session, exists := sm.Sessions[id]

	if exists && session.Finished {
		return nil, false
	}

	return session, exists
}

func (sm *SessionManager) GatherSessions(filter func(*Session) bool, lock bool) map[int]*Session {
	sm.Lock.RLock()
	defer sm.Lock.RUnlock()

	result := make(map[int]*Session)
	for id, session := range sm.Sessions {
		if lock {
			session.Lock.RLock()
		}

		if filter(session) {
			result[id] = session
		}

		if lock {
			session.Lock.RUnlock()
		}
	}
	return result
}

func (sm *SessionManager) FinishSession(session *Session, propagate bool, withNotice bool, lock bool) {
	if lock {
		session.Worker.Lock.Lock()
		defer session.Worker.Lock.Unlock()
	}

	session.Lock.Lock()
	defer session.Lock.Unlock()

	if session.Finished {
		return
	}

	// forcefully cancel interaction
	session.CancelFunc()
	session.Interacting = false

	// whether to send a cancel request to the worker
	if propagate {
		// asynchronously call cancel on the worker
		if withNotice {
			go session.Worker.CancelWithNotice(session.Id)
		} else {
			go session.Worker.Cancel(session.Id)
		}

		sm.controller.Logger.Infof("calling worker %s#%d to cancel session %d", session.Worker.Name, session.Worker.Id, session.Id)
	}

	session.LastUpdate = time.Now()
	if withNotice {
		// if the worker supports cancel_with_notice,
		// the result of the cancel is notified to the controller.
		// we do not mark the session as finished for now
		session.Cancelling = true
	} else {
		// soft delete for now.
		// wait for 10 minutes for a background task to actually delete the session

		// decrement the worker's current session count
		session.Worker.Current--

		// with_cancel is not specified,
		// mark the session as finished
		session.Finished = true
		sm.controller.Logger.Infof("session %d marked as finished", session.Id)
	}
}

func (sm *SessionManager) FinishSessions(filter func(*Session) bool, withNotice bool) {
	sm.Lock.RLock()
	defer sm.Lock.RUnlock()

	for _, session := range sm.Sessions {
		session.Lock.RLock()

		if !session.Finished && filter(session) {
			go sm.FinishSession(session, false, withNotice, true)
		}

		session.Lock.RUnlock()
	}
}

func (sm *SessionManager) CleanSessions() {
	sm.Lock.Lock()
	defer sm.Lock.Unlock()

	var toBeRemoved []int
	finishLine := time.Now().Add(-sm.controller.SessionExpireTime)
	removeLine := time.Now().Add(-sm.controller.SessionRemoveTime)
	for id, session := range sm.Sessions {
		session.Lock.RLock()

		if session.Finished {
			if session.LastUpdate.Before(removeLine) {
				sm.controller.Logger.Infof("session %d removed", id)
				toBeRemoved = append(toBeRemoved, id)
			}
		} else if !session.Interacting && session.LastUpdate.Before(finishLine) {
			sm.controller.Logger.Infof("session %d expired", id)
			go sm.FinishSession(session, true, true, true)
		}

		session.Lock.RUnlock()
	}

	for _, id := range toBeRemoved {
		delete(sm.Sessions, id)
	}
}

func (sm *SessionManager) Dump() map[int]types.SessionData {
	sm.Lock.RLock()
	defer sm.Lock.RUnlock()

	dump := make(map[int]types.SessionData, len(sm.Sessions))
	for id, session := range sm.Sessions {
		session.Lock.RLock()

		if !session.Finished {
			dump[id] = session.Dump()
		}

		session.Lock.RUnlock()
	}
	return dump
}

func (controller *Controller) NewSessionsManager() *SessionManager {
	return &SessionManager{
		controller: controller,
		Sessions:   make(map[int]*Session),
	}
}
