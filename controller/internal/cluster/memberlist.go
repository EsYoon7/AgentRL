package cluster

import (
	"encoding/json"
	"os"

	"github.com/hashicorp/logutils"
	"github.com/hashicorp/memberlist"
	"go.uber.org/zap"
)

type MemberlistOptions struct {
	BindHost string
	BindPort uint16
	Join     []string
	Logger   *zap.SugaredLogger
	NodeInfo *NodeInfo
}

func NewMemberlist(op MemberlistOptions) *NodeRegistry {
	registry := newRegistry()

	config := memberlist.DefaultLANConfig()

	if op.BindHost != "" {
		config.BindAddr = op.BindHost
	}

	if op.BindPort > 0 {
		config.BindPort = int(op.BindPort)
	}

	config.Delegate = &memberlistDelegate{
		logger: op.Logger,
		info:   op.NodeInfo,
	}

	config.Events = &memberlistEvents{
		logger:   op.Logger,
		registry: registry,
	}

	config.LogOutput = &logutils.LevelFilter{
		Levels:   []logutils.LogLevel{"DEBUG", "WARN", "ERROR"},
		MinLevel: "WARN",
		Writer:   os.Stderr,
	}

	ml, err := memberlist.Create(config)
	if err != nil {
		op.Logger.Fatal(err)
	}
	op.Logger.Infof("memberlist listening on %s:%d", config.BindAddr, config.BindPort)

	registry.Memberlist = ml
	op.Logger.Infof("memberlist initialized with node name %s", ml.LocalNode().Name)

	if len(op.Join) > 0 {
		_, err = ml.Join(op.Join)
		if err != nil {
			op.Logger.Fatal(err)
		}
	}

	return registry
}

type memberlistDelegate struct {
	logger *zap.SugaredLogger
	info   *NodeInfo
}

func (d *memberlistDelegate) NodeMeta(limit int) []byte {
	b, err := json.Marshal(d.info)
	if err != nil {
		d.logger.Warnf("failed to marshal node meta: %v", err)
	}

	if len(b) > limit {
		d.logger.Warnf("node meta is too large: %d > %d", len(b), limit)
		return nil
	}

	return b
}

func (*memberlistDelegate) NotifyMsg([]byte) {
	// no-op
}

func (*memberlistDelegate) GetBroadcasts(int, int) [][]byte {
	return nil // no-op
}

func (*memberlistDelegate) LocalState(_ bool) []byte {
	return nil // no-op
}

func (*memberlistDelegate) MergeRemoteState([]byte, bool) {
	// no-op
}

type memberlistEvents struct {
	logger   *zap.SugaredLogger
	registry *NodeRegistry
}

func (e *memberlistEvents) NotifyJoin(n *memberlist.Node) {
	var info *NodeInfo
	err := json.Unmarshal(n.Meta, &info)
	if err == nil {
		info.Address = n.Addr.String()
		e.registry.Set(n.Name, info)
		e.logger.Debugf("node %s joined with address %s", n.Name, info.Address)
	} else {
		e.logger.Warnf("failed to unmarshal node %s meta: %v", n.Name, err)
	}
}

func (e *memberlistEvents) NotifyUpdate(n *memberlist.Node) {
	var info *NodeInfo
	err := json.Unmarshal(n.Meta, &info)
	if err == nil {
		info.Address = n.Addr.String()
		e.registry.Set(n.Name, info)
		e.logger.Debugf("node %s updated address %s", n.Name, info.Address)
	} else {
		e.logger.Warnf("failed to unmarshal node %s meta: %v", n.Name, err)
	}
}

func (e *memberlistEvents) NotifyLeave(n *memberlist.Node) {
	e.registry.Delete(n.Name)
	e.logger.Debugf("node %s left", n.Name)
}
