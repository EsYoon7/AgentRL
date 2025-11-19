package cluster

import (
	"github.com/hashicorp/memberlist"
	"github.com/sasha-s/go-deadlock"
)

type NodeInfo struct {
	Address     string `json:"address"`
	ServicePort uint16 `json:"port"`
}

type NodeRegistry struct {
	mu         deadlock.RWMutex
	data       map[string]*NodeInfo
	Memberlist *memberlist.Memberlist
}

func newRegistry() *NodeRegistry {
	return &NodeRegistry{
		data: make(map[string]*NodeInfo),
	}
}

func (r *NodeRegistry) LocalName() string {
	return r.Memberlist.LocalNode().Name
}

func (r *NodeRegistry) Get(name string) (*NodeInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	info, ok := r.data[name]
	return info, ok
}

func (r *NodeRegistry) Set(name string, info *NodeInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.data[name] = info
}

func (r *NodeRegistry) Delete(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.data, name)
}

func (r *NodeRegistry) Snapshot() map[string]NodeInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	cp := make(map[string]NodeInfo, len(r.data))
	for k, v := range r.data {
		cp[k] = *v
	}

	return cp
}
