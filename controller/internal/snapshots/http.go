package snapshots

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/pb/snapshots_v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

type httpHandler struct {
	server *Server
}

type tagsPayload struct {
	Tags []string `json:"tags"`
}

func (h *httpHandler) RegisterHttpRoutes(e *echo.Echo) {
	e.GET("/api/nodes", h.listNodes)
	e.GET("/api/snapshots", h.listSnapshots)
	e.GET("/api/snapshots/:id", h.getSnapshot)
	e.DELETE("/api/snapshots/:id", h.deleteSnapshot)
	e.POST("/api/snapshots/:id/tags", h.addSnapshotTags)
	e.DELETE("/api/snapshots/:id/tags", h.removeSnapshotTags)
	e.PUT("/api/snapshots/:id/tags", h.setSnapshotTags)
}

func (h *httpHandler) listNodes(c echo.Context) error {
	return c.JSON(http.StatusOK, h.server.manager.NodeRegistry.Snapshot())
}

func (h *httpHandler) listSnapshots(c echo.Context) error {
	var req snapshots_v1.ListSnapshotsRequest

	if v := strings.TrimSpace(c.QueryParam("task_type")); v != "" {
		req.TaskType = proto.String(v)
	}
	if v := strings.TrimSpace(c.QueryParam("task_name")); v != "" {
		req.TaskName = proto.String(v)
	}
	if v := strings.TrimSpace(c.QueryParam("task_index")); v != "" {
		if intVal, err := strconv.ParseInt(v, 10, 64); err == nil {
			req.TaskIndex = &pb.TaskIndex{
				Value: &pb.TaskIndex_IntValue{
					IntValue: intVal,
				},
			}
		} else {
			req.TaskIndex = &pb.TaskIndex{
				Value: &pb.TaskIndex_StringValue{
					StringValue: v,
				},
			}
		}
	}
	if v := strings.TrimSpace(c.QueryParam("env_type")); v != "" {
		req.EnvType = proto.String(v)
	}
	if v := strings.TrimSpace(c.QueryParam("parent_id")); v != "" {
		if _, err := uuid.Parse(v); err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, "invalid parent_id")
		}
		req.ParentId = proto.String(v)
	}
	if v := strings.TrimSpace(c.QueryParam("session_id")); v != "" {
		sessionID, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, "invalid session_id")
		}
		req.SessionId = proto.Int64(sessionID)
	}
	if v := strings.TrimSpace(c.QueryParam("step")); v != "" {
		step, err := strconv.ParseInt(v, 10, 32)
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, "invalid step")
		}
		req.Step = proto.Int32(int32(step))
	}
	if v := strings.TrimSpace(c.QueryParam("page_size")); v != "" {
		pageSize, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, "invalid page_size")
		}
		req.PageSize = proto.Uint32(uint32(pageSize))
	}
	if v := strings.TrimSpace(c.QueryParam("page_token")); v != "" {
		if _, err := uuid.Parse(v); err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, "invalid page_token")
		}
		req.PageToken = proto.String(v)
	}
	if rawTags := c.QueryParams()["tags"]; len(rawTags) > 0 {
		var tags []string
		for _, entry := range rawTags {
			for _, tag := range strings.Split(entry, ",") {
				tags = append(tags, tag)
			}
		}
		if tags = normalizeTags(tags); len(tags) > 0 {
			req.Tags = append(req.Tags, tags...)
		}
	}

	resp, err := h.server.ListSnapshots(c.Request().Context(), &req)
	if err != nil {
		if st, ok := status.FromError(err); ok && st.Code() == codes.NotFound {
			return h.respondProto(c, http.StatusOK, &snapshots_v1.ListSnapshotsResponse{})
		}
		h.server.logger.Errorf("list snapshots failed: %v", err)
		return h.httpErrorFromStatus(err)
	}

	return h.respondProto(c, http.StatusOK, resp)
}

func (h *httpHandler) getSnapshot(c echo.Context) error {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		return echo.NewHTTPError(http.StatusBadRequest, "id is required")
	}
	if _, err := uuid.Parse(id); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid id")
	}

	if v := strings.TrimSpace(c.QueryParam("require_path")); v != "" {
		return echo.NewHTTPError(http.StatusBadRequest, "require_path is not supported via HTTP")
	}

	req := snapshots_v1.GetSnapshotRequest{
		Id: proto.String(id),
	}

	resp, err := h.server.GetSnapshot(c.Request().Context(), &req)
	if err != nil {
		h.server.logger.Errorf("get snapshot %s failed: %v", id, err)
		return h.httpErrorFromStatus(err)
	}

	return h.respondProto(c, http.StatusOK, resp)
}

func (h *httpHandler) deleteSnapshot(c echo.Context) error {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		return echo.NewHTTPError(http.StatusBadRequest, "id is required")
	}
	if _, err := uuid.Parse(id); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid id")
	}

	if _, err := h.server.DeleteSnapshot(c.Request().Context(), &snapshots_v1.DeleteSnapshotRequest{
		Id:        proto.String(id),
		Propagate: proto.Bool(true),
	}); err != nil {
		h.server.logger.Errorf("delete snapshot %s failed: %v", id, err)
		return h.httpErrorFromStatus(err)
	}

	return c.NoContent(http.StatusNoContent)
}

func (h *httpHandler) addSnapshotTags(c echo.Context) error {
	return h.modifySnapshotTags(c, func(ctx echo.Context, id string, tags []string) (proto.Message, error) {
		return h.server.AddSnapshotTags(ctx.Request().Context(), &snapshots_v1.AddSnapshotTagsRequest{
			Id:   proto.String(id),
			Tags: tags,
		})
	})
}

func (h *httpHandler) removeSnapshotTags(c echo.Context) error {
	return h.modifySnapshotTags(c, func(ctx echo.Context, id string, tags []string) (proto.Message, error) {
		return h.server.RemoveSnapshotTags(ctx.Request().Context(), &snapshots_v1.RemoveSnapshotTagsRequest{
			Id:   proto.String(id),
			Tags: tags,
		})
	})
}

func (h *httpHandler) setSnapshotTags(c echo.Context) error {
	return h.modifySnapshotTags(c, func(ctx echo.Context, id string, tags []string) (proto.Message, error) {
		return h.server.SetSnapshotTags(ctx.Request().Context(), &snapshots_v1.SetSnapshotTagsRequest{
			Id:   proto.String(id),
			Tags: tags,
		})
	})
}

type tagsModifier func(ctx echo.Context, id string, tags []string) (proto.Message, error)

func (h *httpHandler) modifySnapshotTags(c echo.Context, fn tagsModifier) error {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		return echo.NewHTTPError(http.StatusBadRequest, "id is required")
	}
	if _, err := uuid.Parse(id); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid id")
	}

	var payload tagsPayload
	if err := c.Bind(&payload); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid payload")
	}
	tags := normalizeTags(payload.Tags)
	if len(tags) == 0 && c.Request().Method != http.MethodPut {
		return echo.NewHTTPError(http.StatusBadRequest, "tags are required")
	}

	resp, err := fn(c, id, tags)
	if err != nil {
		h.server.logger.Errorf("update snapshot %s tags failed: %v", id, err)
		return h.httpErrorFromStatus(err)
	}

	return h.respondProto(c, http.StatusOK, resp)
}

func (h *httpHandler) httpErrorFromStatus(err error) *echo.HTTPError {
	if err == nil {
		return echo.NewHTTPError(http.StatusInternalServerError, "internal error")
	}

	st, ok := status.FromError(err)
	if !ok {
		return echo.NewHTTPError(http.StatusInternalServerError, "internal error")
	}

	switch st.Code() {
	case codes.InvalidArgument:
		return echo.NewHTTPError(http.StatusBadRequest, st.Message())
	case codes.NotFound:
		return echo.NewHTTPError(http.StatusNotFound, st.Message())
	case codes.Unavailable:
		return echo.NewHTTPError(http.StatusServiceUnavailable, st.Message())
	default:
		return echo.NewHTTPError(http.StatusInternalServerError, st.Message())
	}
}

func (h *httpHandler) respondProto(c echo.Context, status int, msg proto.Message) error {
	if msg == nil {
		return c.NoContent(status)
	}

	data, err := protojson.MarshalOptions{
		UseProtoNames:   true,
		EmitUnpopulated: false,
	}.Marshal(msg)
	if err != nil {
		h.server.logger.Errorf("failed to marshal proto response: %v", err)
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to marshal response")
	}

	return c.Blob(status, echo.MIMEApplicationJSON, data)
}
