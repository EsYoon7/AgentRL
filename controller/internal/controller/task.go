package controller

import (
	"fmt"
	"sync/atomic"
	"time"

	"github.com/sasha-s/go-deadlock"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/types"
	"github.com/thudm/agentrl/controller/internal/utils"
	"google.golang.org/grpc"
)

type Task struct {
	nextWorkerId    atomic.Int32
	mapIndicesToInt bool
	Indices         []types.TaskIndex
	Workers         []*Worker
}

func (t *Task) Dump() types.TaskData {
	workers := make(map[int]types.WorkerData, len(t.Workers))
	for _, worker := range t.Workers {
		worker.Lock.RLock()
		workers[worker.Id] = worker.Dump()
		worker.Lock.RUnlock()
	}

	return types.TaskData{
		Indices: t.Indices,
		Workers: workers,
	}
}

type TaskManager struct {
	controller *Controller
	Tasks      map[string]*Task
	Lock       deadlock.RWMutex
}

func (tm *TaskManager) CreateOrValidateTask(name string, indices []types.TaskIndex) (*Task, error) {
	if len(name) == 0 || len(indices) == 0 {
		return nil, fmt.Errorf("task name and indices cannot be empty")
	}

	tm.Lock.Lock()
	defer tm.Lock.Unlock()

	task, exists := tm.Tasks[name]
	if !exists || len(task.Workers) == 0 {
		task = &Task{
			mapIndicesToInt: true,
			Indices:         indices,
			Workers:         make([]*Worker, 0),
		}

		// if all indices in the task are string,
		// create int mapping for them to support the training framework
		for _, index := range indices {
			val, ok := index.Value.(int)
			if ok && val != -1 {
				// this index is int and is not custom
				// int mapping is not needed then
				task.mapIndicesToInt = false
				break
			}
		}

		tm.Tasks[name] = task
		tm.controller.Logger.Infof("task %s registered with %d indices", name, len(indices))
	} else {
		// validate if indices are the same
		if !utils.CompareTaskIndices(task.Indices, indices) {
			tm.controller.Logger.Warnf("task %s has inconsistent indices", name)
			return nil, fmt.Errorf("task %s already exists with different indices", name)
		}
	}

	return task, nil
}

func (tm *TaskManager) UpdateWorker(name string, address string, capacity int, stream *grpc.BidiStreamingServer[pb.WorkerStreamEnvelope, pb.WorkerStreamEnvelope]) (*Worker, error) {
	if len(address) == 0 {
		return nil, fmt.Errorf("worker address cannot be empty")
	}

	if capacity <= 0 {
		return nil, fmt.Errorf("worker capacity must be a positive integer")
	}

	tm.Lock.Lock()

	task, exists := tm.Tasks[name]
	if !exists {
		tm.Lock.Unlock()
		return nil, fmt.Errorf("task %s does not exist", name)
	}

	var worker *Worker
	for _, w := range task.Workers {
		if w.Address == address {
			worker = w
			break
		}
	}

	if worker == nil {
		// if there are workers in other tasks with the same address,
		// then it's already stopped and addresses have changed
		for _, t := range tm.Tasks {
			if t == task {
				continue
			}
			for _, w := range t.Workers {
				if w.Address == address {
					tm.controller.Logger.Warnf("worker %s#%d with same address %s already exists, removing it", w.Name, w.Id, address)
					go tm.RemoveWorker(w)
					go tm.controller.SessionManager.FinishSessions(func(session *Session) bool {
						return session.Worker == w
					}, false)
				}
			}
		}

		// worker registration
		workerId := int(task.nextWorkerId.Add(1))
		worker = &Worker{
			controller: tm.controller,
			stream:     stream,
			pendingRpc: make(map[string]*grpcRequest),
			Id:         workerId,
			Name:       name,
			Address:    address,
			Capacity:   capacity,
			LastVisit:  time.Now(),
			Stale:      false,
			Status:     WorkerStatusAlive,
		}

		task.Workers = append(task.Workers, worker)
		go worker.Sync()

		tm.controller.Logger.Infof("worker %s#%d registered", name, workerId)
		tm.Lock.Unlock()
	} else {
		worker.Lock.Lock()
		tm.Lock.Unlock() // handle global lock
		defer worker.Lock.Unlock()

		worker.Capacity = capacity
		worker.LastVisit = time.Now()
		if stream != nil && stream != worker.stream {
			worker.stream = stream
			tm.controller.Logger.Infof("worker %s#%d connected with new gRPC stream", name, worker.Id)
		}

		if worker.Status != WorkerStatusAlive {
			// worker has lost contact for a while, we need to sync with it
			tm.controller.Logger.Infof("worker %s#%d status abnormal, syncing...", name, worker.Id)
			go worker.Sync()
		}
		worker.Status = WorkerStatusAlive
	}

	return worker, nil
}

func (tm *TaskManager) MarkWorkerStale(name string, id int, stale bool) error {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	task, exists := tm.Tasks[name]
	if !exists {
		return fmt.Errorf("task %s does not exist", name)
	}

	for _, worker := range task.Workers {
		worker.Lock.Lock()
		if worker.Id == id {

			worker.Stale = stale
			tm.controller.Logger.Infof("worker %s#%d marked as stale", name, id)
			worker.Lock.Unlock()
			return nil
		}
		worker.Lock.Unlock()
	}

	return fmt.Errorf("worker %s#%d does not exist", name, id)
}

func (tm *TaskManager) DispatchTask(name string, index types.TaskIndex, customTask interface{}) (*Session, error) {
	if index.IsCustom() {
		if customTask == nil {
			return nil, fmt.Errorf("no custom task provided")
		}
	} else if customTask != nil {
		tm.controller.Logger.Warnf("Ignoring custom task for non-custom index: %s", index)
		customTask = nil
	}

	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	task, exists := tm.Tasks[name]
	if !exists {
		return nil, fmt.Errorf("task %s does not exist", name)
	}

	if task.mapIndicesToInt {
		// try to map the int index to original index
		intIndex, err := index.Int()
		if err == nil && intIndex >= 0 && intIndex < len(task.Indices) {
			index = task.Indices[intIndex]
		}
		// if fails, keep the index as is
	}

	if !utils.ContainsTaskIndex(task.Indices, index) {
		return nil, fmt.Errorf("task %s does not support index %s", name, index)
	}

	for {
		var targetWorker *Worker

		maxAvailable := 0
		for _, worker := range task.Workers {
			worker.Lock.Lock()

			if worker.Status != WorkerStatusAlive || worker.Stale {
				worker.Lock.Unlock()
				continue
			}

			if time.Since(worker.LastVisit) > tm.controller.HeartbeatTimeout {
				worker.Status = WorkerStatusComa
				tm.controller.Logger.Warnf("Worker %s#%d status changed to COMA", worker.Name, worker.Id)
				worker.Lock.Unlock()
				continue
			}

			available := worker.Capacity - worker.Current

			// to avoid deadlock, we unlock the worker inside the loop.
			// this might bring slight possibility of inconsistency,
			// and if later check, we can retry the dispatching process
			// until no idle worker can be chosen.
			worker.Lock.Unlock()

			if available > maxAvailable {
				maxAvailable = available
				targetWorker = worker
				continue
			}
		}

		if targetWorker == nil {
			return nil, fmt.Errorf("no workers available for task %s", name)
		}

		targetWorker.Lock.Lock()

		if targetWorker.Status != WorkerStatusAlive || targetWorker.Current >= targetWorker.Capacity {
			// state changed after we checked, retry
			targetWorker.Lock.Unlock()
			continue
		}

		targetWorker.Current++
		tm.controller.Logger.Infof("Selected worker %d for task %s, load: %d/%d", targetWorker.Id, targetWorker.Name, targetWorker.Current, targetWorker.Capacity)

		targetWorker.Lock.Unlock()

		session := tm.controller.SessionManager.CreateSession(targetWorker, index, customTask)
		return session, nil
	}
}

func (tm *TaskManager) CallCancelAll() {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	// clear local sessions
	tm.controller.SessionManager.FinishSessions(func(session *Session) bool {
		return true
	}, true)

	for _, task := range tm.Tasks {
		for _, worker := range task.Workers {
			worker.Lock.RLock()

			if worker.Status != WorkerStatusDead {
				worker.Lock.RUnlock()

				// call worker cancel all
				go worker.CancelAllWithNotice(false)
			} else {
				worker.Lock.RUnlock()
			}
		}
	}
}

func (tm *TaskManager) CallCancelAllFor(taskName string) {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	// clear local sessions for the worker
	tm.controller.SessionManager.FinishSessions(func(session *Session) bool {
		return session.Worker.Name == taskName
	}, true)

	task, exists := tm.Tasks[taskName]
	if !exists {
		return
	}

	for _, worker := range task.Workers {
		worker.Lock.RLock()

		if worker.Status != WorkerStatusDead {
			worker.Lock.RUnlock()

			// call worker cancel all
			go worker.CancelAllWithNotice(false)
		} else {
			worker.Lock.RUnlock()
		}
	}
}

func (tm *TaskManager) CallSyncAll() {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	for _, task := range tm.Tasks {
		for _, worker := range task.Workers {
			go func() {
				// random delay to avoid all workers syncing at the same time
				delay := time.Duration(utils.RandInt(0, 2000)) * time.Millisecond
				time.Sleep(delay)
				worker.Sync()
			}()
		}
	}
}

func (tm *TaskManager) RemoveWorker(worker *Worker) {
	tm.Lock.Lock()
	defer tm.Lock.Unlock()

	task, exists := tm.Tasks[worker.Name]
	if !exists {
		// does not exist
		return
	}

	worker.Lock.Lock()
	defer worker.Lock.Unlock()

	if worker.Status == WorkerStatusDead {
		// does not exist
		return
	}

	// mark worker as dead and delete it from the task
	worker.Status = WorkerStatusDead
	for i, w := range task.Workers {
		if w == worker {
			task.Workers = append(task.Workers[:i], task.Workers[i+1:]...)
			break
		}
	}
	tm.controller.Logger.Infof("Worker %s#%d removed from task %s", worker.Name, worker.Id, worker.Name)

	// if the task has no workers left, remove the task
	if len(task.Workers) == 0 {
		delete(tm.Tasks, worker.Name)
		tm.controller.Logger.Infof("Task %s removed", worker.Name)
	}
}

func (tm *TaskManager) CleanWorkers() {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	comaLine := time.Now().Add(-tm.controller.HeartbeatTimeout)
	removeLine := time.Now().Add(-tm.controller.WorkerRemoveTime)
	for _, task := range tm.Tasks {
		for _, worker := range task.Workers {
			worker.Lock.Lock()

			if worker.Status == WorkerStatusAlive && worker.LastVisit.Before(comaLine) {
				// worker is coma
				worker.Status = WorkerStatusComa
				tm.controller.Logger.Warnf("Worker %s#%d status changed to COMA", worker.Name, worker.Id)
			}

			if worker.Current == 0 && worker.Status == WorkerStatusComa && worker.LastVisit.Before(removeLine) {
				// worker should be removed
				go tm.RemoveWorker(worker)
				tm.controller.Logger.Infof("Worker %s#%d removed", worker.Name, worker.Id)
			}

			worker.Lock.Unlock()
		}
	}
}

func (tm *TaskManager) DumpIndices(name string) ([]types.TaskIndex, error) {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	task, exists := tm.Tasks[name]
	if !exists {
		return nil, fmt.Errorf("task %s does not exist", name)
	}

	return task.Indices, nil
}

func (tm *TaskManager) Dump() types.ListWorkersResponse {
	tm.Lock.RLock()
	defer tm.Lock.RUnlock()

	tasks := make(types.ListWorkersResponse, len(tm.Tasks))
	for name, task := range tm.Tasks {
		tasks[name] = task.Dump()
	}
	return tasks
}

func (controller *Controller) NewTaskManager() *TaskManager {
	return &TaskManager{
		controller: controller,
		Tasks:      make(map[string]*Task),
	}
}
