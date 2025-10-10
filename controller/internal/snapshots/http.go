package snapshots

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/labstack/echo/v4"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/pb/snapshots_v1"
	"google.golang.org/protobuf/proto"
)

type httpHandler struct {
	server *Server
}

func (h *httpHandler) RegisterHttpRoutes(e *echo.Echo) {
	e.GET("/api/nodes", h.listNodes)
	e.GET("/api/snapshots", h.listSnapshots)
	e.GET("/api/snapshots/:id", h.getSnapshot)
	e.DELETE("/api/snapshots/:id", h.deleteSnapshot)
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

	resp, err := h.server.ListSnapshots(c.Request().Context(), &req)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return c.JSON(http.StatusOK, &snapshots_v1.ListSnapshotsResponse{})
		}
		h.server.logger.Errorf("list snapshots failed: %v", err)
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to list snapshots")
	}

	return c.JSON(http.StatusOK, resp)
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
		if errors.Is(err, pgx.ErrNoRows) {
			return echo.NewHTTPError(http.StatusNotFound, "snapshot not found")
		}
		h.server.logger.Errorf("get snapshot %s failed: %v", id, err)
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to get snapshot")
	}

	return c.JSON(http.StatusOK, resp)
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
		return echo.NewHTTPError(http.StatusInternalServerError, "failed to delete snapshot")
	}

	return c.NoContent(http.StatusNoContent)
}
