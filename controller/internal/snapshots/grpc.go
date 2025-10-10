package snapshots

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/google/uuid"
	"github.com/thudm/agentrl/controller/internal/pb"
	"github.com/thudm/agentrl/controller/internal/pb/snapshots_v1"
	"github.com/thudm/agentrl/controller/internal/types"
	"github.com/thudm/agentrl/controller/internal/utils"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Server struct {
	logger  *zap.SugaredLogger
	manager *Manager
	snapshots_v1.UnimplementedSnapshotsManagerServer
}

func (s *Server) convertTaskIndex(index *pb.TaskIndex) types.NullTaskIndex {
	if index == nil {
		return types.NullTaskIndex{}
	}

	if val, ok := index.Value.(*pb.TaskIndex_IntValue); ok {
		return types.NullTaskIndex{
			TaskIndex: types.TaskIndex{
				Value: int(val.IntValue),
			},
			Valid: true,
		}
	}

	if val, ok := index.Value.(*pb.TaskIndex_StringValue); ok {
		return types.NullTaskIndex{
			TaskIndex: types.TaskIndex{
				Value: val.StringValue,
			},
			Valid: true,
		}
	}

	return types.NullTaskIndex{}
}

func (s *Server) convertDBRecord(record *DatabaseRecord) *snapshots_v1.Snapshot {
	snapshot := &snapshots_v1.Snapshot{
		Id: proto.String(record.ID.String()),
	}

	if record.ParentID.Valid {
		snapshot.ParentId = proto.String(record.ParentID.UUID.String())
	}

	hierarchy := strings.Split(record.Hierarchy, ".")
	snapshot.Hierarchy = hierarchy

	if record.TaskType.Valid {
		snapshot.TaskType = &record.TaskType.String
	}

	if record.TaskName.Valid {
		snapshot.TaskName = &record.TaskName.String
	}

	if record.TaskIndex.Valid {
		intVal, err := record.TaskIndex.TaskIndex.Int()
		if err == nil {
			snapshot.TaskIndex = &pb.TaskIndex{
				Value: &pb.TaskIndex_IntValue{
					IntValue: int64(intVal),
				},
			}
		} else {
			snapshot.TaskIndex = &pb.TaskIndex{
				Value: &pb.TaskIndex_StringValue{
					StringValue: record.TaskIndex.TaskIndex.String(),
				},
			}
		}
	}

	if record.EnvType.Valid {
		snapshot.EnvType = &record.EnvType.String
	}

	if record.SessionID.Valid {
		snapshot.SessionId = &record.SessionID.Int64
	}

	if record.Step.Valid {
		snapshot.Step = &record.Step.Int32
	}

	snapshot.Node = proto.String(record.Node)
	snapshot.CreatedAt = timestamppb.New(record.CreatedAt)

	return snapshot
}

func (s *Server) GetStorePath(_ context.Context, _ *emptypb.Empty) (*snapshots_v1.GetStorePathResponse, error) {
	path := s.manager.Store.RootPath
	total, free, err := utils.DiskUsage(path)
	if err != nil {
		s.logger.Warnf("failed to get disk usage for path %s: %v", path, err)
	}

	return &snapshots_v1.GetStorePathResponse{
		RootPath:   &path,
		TotalBytes: &total,
		FreeBytes:  &free,
	}, nil
}

func (s *Server) CreateSnapshot(ctx context.Context, request *snapshots_v1.CreateSnapshotRequest) (*snapshots_v1.CreateSnapshotResponse, error) {
	record := DatabaseRecord{
		TaskType: sql.NullString{
			String: request.GetTaskType(),
			Valid:  request.TaskType != nil,
		},
		TaskName: sql.NullString{
			String: request.GetTaskName(),
			Valid:  request.TaskName != nil,
		},
		TaskIndex: s.convertTaskIndex(request.TaskIndex),
		EnvType: sql.NullString{
			String: request.GetEnvType(),
			Valid:  request.EnvType != nil,
		},
		SessionID: sql.NullInt64{
			Int64: request.GetSessionId(),
			Valid: request.SessionId != nil,
		},
		Step: sql.NullInt32{
			Int32: request.GetStep(),
			Valid: request.Step != nil,
		},
		Node: s.manager.NodeRegistry.LocalName(),
	}

	if request.ParentId != nil {
		parentId, err := uuid.Parse(request.GetParentId())
		if err != nil {
			return nil, fmt.Errorf("invalid parent_id: %w", err)
		}
		record.ParentID = uuid.NullUUID{
			UUID:  parentId,
			Valid: true,
		}
	}

	id, err := s.manager.Database.CreateSnapshot(ctx, &record)
	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot record: %w", err)
	}

	path, err := s.manager.Store.CreateTemporary(id, request.GetExpectedSize())
	if err != nil {
		err1 := s.manager.Database.DeleteSnapshot(ctx, id)
		if err1 != nil {
			s.logger.Warnf("failed to delete snapshot record after store creation failure: %v", err1)
		}
		return nil, fmt.Errorf("failed to create snapshot store: %w", err)
	}

	return &snapshots_v1.CreateSnapshotResponse{
		Id:   &id,
		Path: &path,
	}, nil
}

func (s *Server) MarkReady(ctx context.Context, request *snapshots_v1.MarkReadyRequest) (*emptypb.Empty, error) {
	if request.GetId() == "" {
		return nil, fmt.Errorf("id is required")
	}

	size, err := s.manager.Store.SaveTemporary(request.GetId(), false)
	if err != nil {
		return nil, fmt.Errorf("failed to save snapshot: %w", err)
	}

	if err = s.manager.Database.SetSnapshotSize(ctx, request.GetId(), size); err != nil {
		s.manager.Store.DeleteSnapshot(request.GetId())
		err1 := s.manager.Database.DeleteSnapshot(ctx, request.GetId())
		if err1 != nil {
			s.logger.Warnf("failed to delete snapshot record after database size update failure: %v", err1)
		}
		return nil, fmt.Errorf("failed to set snapshot size: %w", err)
	}

	return &emptypb.Empty{}, nil
}

func (s *Server) ListSnapshots(ctx context.Context, request *snapshots_v1.ListSnapshotsRequest) (*snapshots_v1.ListSnapshotsResponse, error) {
	example := DatabaseRecord{
		TaskType: sql.NullString{
			String: request.GetTaskType(),
			Valid:  request.TaskType != nil,
		},
		TaskName: sql.NullString{
			String: request.GetTaskName(),
			Valid:  request.TaskName != nil,
		},
		TaskIndex: s.convertTaskIndex(request.TaskIndex),
		EnvType: sql.NullString{
			String: request.GetEnvType(),
			Valid:  request.EnvType != nil,
		},
		SessionID: sql.NullInt64{
			Int64: request.GetSessionId(),
			Valid: request.SessionId != nil,
		},
		Step: sql.NullInt32{
			Int32: request.GetStep(),
			Valid: request.Step != nil,
		},
	}

	if request.PageToken != nil {
		pageToken, err := uuid.Parse(request.GetPageToken())
		if err != nil {
			return nil, fmt.Errorf("invalid page_token: %w", err)
		}
		example.ID = pageToken
	}

	if request.ParentId != nil {
		parentId, err := uuid.Parse(request.GetParentId())
		if err != nil {
			return nil, fmt.Errorf("invalid parent_id: %w", err)
		}
		example.ParentID = uuid.NullUUID{
			UUID:  parentId,
			Valid: true,
		}
	}

	res, err := s.manager.Database.ListSnapshots(ctx, &example, int(request.GetPageSize()))
	if err != nil {
		return nil, fmt.Errorf("failed to list snapshots: %w", err)
	}

	resp := &snapshots_v1.ListSnapshotsResponse{
		Snapshots: make([]*snapshots_v1.Snapshot, len(res)),
	}
	for i, record := range res {
		resp.Snapshots[i] = s.convertDBRecord(record)
	}
	resp.PreviousPageToken = request.PageToken
	if len(res) > 0 && len(res) == int(request.GetPageSize()) {
		resp.NextPageToken = proto.String(res[len(res)-1].ID.String())
	}

	return resp, nil
}

func (s *Server) GetSnapshot(ctx context.Context, request *snapshots_v1.GetSnapshotRequest) (*snapshots_v1.GetSnapshotResponse, error) {
	record, err := s.manager.Database.GetSnapshot(ctx, request.GetId())
	if err != nil {
		return nil, fmt.Errorf("failed to get snapshot: %w", err)
	}

	resp := &snapshots_v1.GetSnapshotResponse{
		Snapshot: s.convertDBRecord(record),
	}

	if request.GetRequirePath() {
		path := s.manager.Store.CheckData(record.ID.String())
		if path == "" {
			node := record.Node
			if node == "" || node == s.manager.NodeRegistry.LocalName() {
				return nil, fmt.Errorf("snapshot data not found on local node")
			}

			info, ok := s.manager.NodeRegistry.Get(node)
			if !ok {
				return nil, fmt.Errorf("node %s not found in registry", node)
			}

			ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
			defer cancel()

			addr := net.JoinHostPort(info.Address, strconv.Itoa(int(info.ServicePort)))
			conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				return nil, fmt.Errorf("failed to connect to node %s at %s: %w", node, addr, err)
			}
			defer conn.Close()

			client := snapshots_v1.NewSnapshotsManagerClient(conn)
			stream, err := client.StreamArchive(ctx, &snapshots_v1.StreamArchiveRequest{
				Id: request.Id,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to start archive stream from node %s at %s: %w", node, addr, err)
			}

			// after connecting to the remote node, create a temporary directory for storing the snapshot
			path, err = s.manager.Store.CreateTemporary(record.ID.String(), uint64(record.Size.Int64))
			if err != nil {
				return nil, fmt.Errorf("failed to create temporary store for snapshot: %w", err)
			}

			pr, pw := io.Pipe()
			hasher := sha256.New()
			counter := utils.NewCountWriter()
			tee := io.TeeReader(pr, io.MultiWriter(hasher, counter))

			errCh := make(chan error, 1)
			go func() {
				errCh <- s.manager.Store.ExtractArchive(ctx, record.ID.String(), tee)
				close(errCh)
			}()

			var eof *snapshots_v1.ArchiveChunk_EOF
			var finalErr error

			for {
				msg, err := stream.Recv()
				if err == io.EOF {
					_ = pw.Close()
					finalErr = fmt.Errorf("stream ended without EOF metadata")
					break
				}
				if err != nil {
					_ = pw.CloseWithError(err)
					finalErr = err
					break
				}

				switch p := msg.Payload.(type) {
				case *snapshots_v1.ArchiveChunk_Data:
					if len(p.Data) == 0 {
						continue
					}

					if _, err := pw.Write(p.Data); err != nil {
						_ = pw.CloseWithError(err)
						finalErr = err
						break
					}

				case *snapshots_v1.ArchiveChunk_Eof:
					eof = p.Eof
					// Close writer so the store completes.
					_ = pw.Close()
					finalErr = nil
					break

				default:
					err = fmt.Errorf("unknown payload in stream")
					_ = pw.CloseWithError(err)
					finalErr = err
					break
				}
				if finalErr != nil || eof != nil {
					break
				}
			}

			if finalErr == nil {
				// Wait for the store to finish extracting.
				finalErr = <-errCh
			}
			if finalErr == nil && eof == nil {
				finalErr = fmt.Errorf("missing EOF metadata from server")
			}
			if finalErr == nil {
				sumHex := hex.EncodeToString(hasher.Sum(nil))
				if sumHex != eof.GetSha256Tar() {
					finalErr = fmt.Errorf("sha256 mismatch: got %s, expected %s", sumHex, eof.GetSha256Tar())
				}
			}
			if finalErr == nil {
				total := counter.Count()
				if total != eof.GetTotalSize() {
					finalErr = fmt.Errorf("size mismatch: got %d, expected %d", total, eof.GetTotalSize())
				}
			}

			// commit
			if finalErr == nil {
				_, finalErr = s.manager.Store.SaveTemporary(record.ID.String(), true)
			}

			if finalErr != nil {
				// clean up
				s.manager.Store.DeleteSnapshot(record.ID.String())
				return nil, finalErr
			}
		}

		resp.Path = &path
	}

	return resp, nil
}

func (s *Server) DeleteSnapshot(ctx context.Context, request *snapshots_v1.DeleteSnapshotRequest) (*emptypb.Empty, error) {
	if request.GetId() == "" {
		return nil, fmt.Errorf("id is required")
	}

	s.manager.Store.DeleteSnapshot(request.GetId())

	if request.GetPropagate() {
		err := s.manager.Database.DeleteSnapshot(ctx, request.GetId())
		if err != nil {
			return nil, fmt.Errorf("failed to delete snapshot record: %w", err)
		}

		for name, info := range s.manager.NodeRegistry.Snapshot() {
			if name == s.manager.NodeRegistry.LocalName() {
				continue
			}

			go func(name, addr string) {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				defer cancel()

				conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
				if err != nil {
					s.logger.Errorf("failed to connect to node %s at %s: %v", name, addr, err)
					return
				}
				defer conn.Close()

				client := snapshots_v1.NewSnapshotsManagerClient(conn)
				propagate := false
				if _, err = client.DeleteSnapshot(ctx, &snapshots_v1.DeleteSnapshotRequest{
					Id:        request.Id,
					Propagate: &propagate,
				}); err != nil {
					s.logger.Errorf("failed to propagate snapshot deletion to node %s at %s: %v", name, addr, err)
				}
			}(name, net.JoinHostPort(info.Address, strconv.Itoa(int(info.ServicePort))))
		}
	}

	return &emptypb.Empty{}, nil
}

func (s *Server) StreamArchive(request *snapshots_v1.StreamArchiveRequest, stream grpc.ServerStreamingServer[snapshots_v1.ArchiveChunk]) error {
	if request.GetId() == "" {
		return fmt.Errorf("id is required")
	}

	ctx, cancel := context.WithTimeout(stream.Context(), 10*time.Minute)
	defer cancel()

	r, err := s.manager.Store.StreamArchive(ctx, request.GetId())
	if err != nil {
		return fmt.Errorf("failed to stream archive: %w", err)
	}
	defer r.Close()

	chunkSize := 1 << 20 // 1 MB
	buf := make([]byte, chunkSize)
	hasher := sha256.New()
	counter := utils.NewCountWriter()
	tee := io.TeeReader(r, io.MultiWriter(hasher, counter))

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, err := io.ReadFull(tee, buf)
		if n > 0 {
			// Send chunk. It's safe to reuse buf across sends because gRPC marshals the message before Send returns.
			err = stream.Send(&snapshots_v1.ArchiveChunk{
				Payload: &snapshots_v1.ArchiveChunk_Data{
					Data: buf[:n],
				},
			})
		}

		if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		}
		if err != nil {
			return err
		}
	}

	return stream.Send(&snapshots_v1.ArchiveChunk{
		Payload: &snapshots_v1.ArchiveChunk_Eof{
			Eof: &snapshots_v1.ArchiveChunk_EOF{
				TotalSize: proto.Uint64(counter.Count()),
				Sha256Tar: proto.String(hex.EncodeToString(hasher.Sum(nil))),
			},
		},
	})
}
