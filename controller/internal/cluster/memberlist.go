package cluster

import (
	"github.com/hashicorp/memberlist"
	"go.uber.org/zap"
)

type MemberlistOptions struct {
	BindHost string
	BindPort uint16
	Join     []string
	Logger   *zap.SugaredLogger
}

func CreateMemberlist(op *MemberlistOptions) *memberlist.Memberlist {
	config := memberlist.DefaultLANConfig()

	if op.BindHost != "" {
		config.BindAddr = op.BindHost
	}

	if op.BindPort > 0 {
		config.BindPort = int(op.BindPort)
	}

	ml, err := memberlist.Create(config)
	if err != nil {
		op.Logger.Fatal(err)
	}

	op.Logger.Infof("memberlist listening on %s:%d", config.BindAddr, config.BindPort)
	op.Logger.Infof("memberlist initialized with node name %s", ml.LocalNode().Name)

	if len(op.Join) > 0 {
		n, err := ml.Join(op.Join)
		if err != nil {
			op.Logger.Fatal(err)
		}

		op.Logger.Infof("joined %d other nodes", n)
	}

	return ml
}
