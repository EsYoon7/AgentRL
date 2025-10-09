package types

type WorkerData struct {
	Id        int    `json:"id"`
	Address   string `json:"address"`
	Capacity  int    `json:"capacity"`
	Current   int    `json:"current"`
	LastVisit string `json:"last_visit"`
	Status    string `json:"status"`
	Stale     bool   `json:"stale"`
	Locked    bool   `json:"locked"`
}

type TaskData struct {
	Indices []TaskIndex        `json:"indices"`
	Workers map[int]WorkerData `json:"workers"`
}

type ListWorkersResponse map[string]TaskData

type SessionData struct {
	Name       string      `json:"name"`
	Index      TaskIndex   `json:"index"`
	CustomTask interface{} `json:"custom_task"`
	Start      string      `json:"start"`
	LastUpdate string      `json:"last_update"`
	WorkerId   int         `json:"worker_id"`
	Locked     bool        `json:"locked"`
	Cancelling bool        `json:"cancelling"`
}

type ListSessionsResponse map[int]SessionData
