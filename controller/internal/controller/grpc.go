package controller

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/labstack/echo/v4"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/pb/controller_v1"
	"github.com/thudm/agentrl/controller/internal/types"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type GrpcServer struct {
	controller *Controller
	controller_v1.UnimplementedControllerServer
}

type grpcRequest struct {
	respCh chan<- *controller_v1.WorkerStreamEnvelope
}

func (g *GrpcServer) convertTaskIndex(index *pb.TaskIndex) types.TaskIndex {
	if index == nil {
		return types.TaskIndex{}
	}

	if val, ok := index.Value.(*pb.TaskIndex_IntValue); ok {
		return types.TaskIndex{
			Value: int(val.IntValue),
		}
	}

	if val, ok := index.Value.(*pb.TaskIndex_StringValue); ok {
		return types.TaskIndex{
			Value: val.StringValue,
		}
	}

	return types.TaskIndex{}
}

func (g *GrpcServer) workerResponse(requestId string, err error, data interface{}) *controller_v1.WorkerStreamEnvelope {
	var respCode int32 = http.StatusOK
	respMessage := ""

	if err != nil {
		var httpErr *echo.HTTPError
		if errors.As(err, &httpErr) {
			respCode = int32(httpErr.Code)
			message, ok := httpErr.Message.(string)
			if ok {
				respMessage = message
			}
		} else {
			respCode = http.StatusInternalServerError
		}

		if respMessage == "" {
			respMessage = err.Error()
		}
	}

	resp := &controller_v1.WorkerStreamEnvelope_WorkerResponse{
		Code:    &respCode,
		Message: &respMessage,
		Json:    nil,
	}

	if data != nil {
		jsonBytes, err := json.Marshal(data)
		if err != nil {
			return g.workerResponse(requestId, err, nil)
		}
		resp.Json = jsonBytes
	}

	respType := controller_v1.WorkerStreamEnvelope_RESPONSE
	return &controller_v1.WorkerStreamEnvelope{
		Id:        &requestId,
		Type:      &respType,
		Timestamp: timestamppb.Now(),
		Body: &controller_v1.WorkerStreamEnvelope_WorkerResponse_{
			WorkerResponse: resp,
		},
	}
}

func (g *GrpcServer) WorkerStream(stream grpc.BidiStreamingServer[controller_v1.WorkerStreamEnvelope, controller_v1.WorkerStreamEnvelope]) error {
	var worker *Worker = nil

	defer func() {
		// when the stream is closed, the worker should be considered unhealthy
		if worker != nil {
			g.controller.Logger.Warnf("worker %s#%d disconnected, putting it into coma", worker.Name, worker.Id)
			worker.SetStatus(WorkerStatusComa)
		}
	}()

	g.controller.Logger.Debugf("worker stream connected: %s", stream)

	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		requestId := in.GetId()
		requestType := in.GetType()

		switch requestType {
		case controller_v1.WorkerStreamEnvelope_HEARTBEAT:
			request := in.GetReceiveHeartbeatRequest()
			if request == nil {
				g.controller.Logger.Errorf("failed to parse heartbeat request: %v", in)
				continue
			}

			indices := make([]types.TaskIndex, 0, len(request.Indices))
			for _, index := range request.Indices {
				indices = append(indices, g.convertTaskIndex(index))
			}

			_, err = g.controller.TaskManager.CreateOrValidateTask(request.GetName(), indices)
			if err != nil {
				g.controller.Logger.Errorf("failed to create or validate task: %v", err)
				continue
			}

			fakeAddress := "grpc://" + request.GetId()
			worker, err = g.controller.TaskManager.UpdateWorker(request.GetName(), fakeAddress, int(request.GetConcurrency()), &stream)
			if err != nil {
				g.controller.Logger.Errorf("failed to update worker: %v", err)
				continue
			}

		case controller_v1.WorkerStreamEnvelope_REQUEST:
			if message := in.GetSessionCancelNotice(); message != nil {
				sessionId := int(message.GetSessionId())
				err = g.controller.handleCancelNoticeGeneric(sessionId)
				out := g.workerResponse(requestId, err, nil)
				if err = stream.Send(out); err != nil {
					g.controller.Logger.Errorf("failed to send gRPC response: %v", err)
				}
			}

		case controller_v1.WorkerStreamEnvelope_RESPONSE:
			if worker != nil {
				worker.FinalizeGrpcCall(requestId, in)
			}
		}
	}
}
