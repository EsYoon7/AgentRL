package snapshots

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/thudm/agentrl/controller/internal/types"
	"go.uber.org/zap"
)

var migration = []string{
	`CREATE EXTENSION IF NOT EXISTS ltree`,
	`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`,
	`CREATE TABLE IF NOT EXISTS snapshots
	(
		id         uuid PRIMARY KEY,
		parent_id  uuid,
		hierarchy  ltree,
		task_type  text,
		task_name  text,
		task_index text,
		env_type   text,
		session_id bigint,
		step       integer,
		node       text NOT NULL,
		size       bigint,
		created_at timestamptz NOT NULL DEFAULT now(),
		CONSTRAINT fk_parent FOREIGN KEY(parent_id) REFERENCES snapshots(id)
	)`,
	`CREATE TABLE IF NOT EXISTS snapshot_tags
	(
		snapshot_id uuid NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
		tag         text NOT NULL,
		created_at  timestamptz NOT NULL DEFAULT now(),
		PRIMARY KEY (snapshot_id, tag)
	)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_hierarchy ON snapshots USING GIST (hierarchy)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_task_type ON snapshots USING HASH (task_type)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_task_name ON snapshots USING HASH (task_name)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_task_index ON snapshots (task_index)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_env_type ON snapshots USING HASH (env_type)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_session_id ON snapshots (session_id)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_step ON snapshots (step)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshots_node ON snapshots USING HASH (node)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshot_tags_tag ON snapshot_tags USING HASH (tag)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshot_tags_snapshot ON snapshot_tags (snapshot_id)`,
	`CREATE INDEX IF NOT EXISTS idx_snapshot_tags_tag_snapshot ON snapshot_tags (tag, snapshot_id)`,
}

var migrationAdvisoryLockID int64 = 660465433444466964

type DatabaseRecord struct {
	ID        uuid.UUID
	ParentID  uuid.NullUUID
	Hierarchy string
	TaskType  sql.NullString
	TaskName  sql.NullString
	TaskIndex types.NullTaskIndex
	EnvType   sql.NullString
	SessionID sql.NullInt64
	Step      sql.NullInt32
	Node      string
	Size      sql.NullInt64
	CreatedAt time.Time
	Tags      []string
}

type Database struct {
	logger *zap.SugaredLogger
	conn   *pgxpool.Pool
}

func NewDatabase(logger *zap.SugaredLogger, connection string) *Database {
	conn, err := pgxpool.New(context.Background(), connection)
	if err != nil {
		logger.Fatalf("failed to connect to database: %v", err)
	}

	db := &Database{
		logger: logger,
		conn:   conn,
	}

	if err = db.migrate(); err != nil {
		logger.Fatalf("failed to migrate database: %v", err)
	}

	return db
}

// CreateSnapshot creates a new snapshot record in the database, generating a new ID and returns it.
func (db *Database) CreateSnapshot(ctx context.Context, record *DatabaseRecord) (string, error) {
	if record.ID == uuid.Nil {
		newId, err := uuid.NewV6()
		if err != nil {
			return "", err
		}
		record.ID = newId
	}

	err := db.withTx(ctx, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `
			INSERT INTO snapshots (id, hierarchy, task_type, task_name, task_index, env_type, session_id, step, node)
			VALUES ($1::uuid, $1::text::ltree, $2, $3, $4, $5, $6, $7, $8)
		`, record.ID, record.TaskType, record.TaskName, record.TaskIndex, record.EnvType, record.SessionID, record.Step, record.Node); err != nil {
			return err
		}

		if record.ParentID.Valid {
			if _, err := tx.Exec(ctx, `
				UPDATE snapshots SET parent_id = $2, hierarchy = (
					(SELECT hierarchy FROM snapshots WHERE id = $2)::text || '.' || $1::text
				)::ltree WHERE id = $1
			`, record.ID, record.ParentID); err != nil {
				return err
			}
		}

		if len(record.Tags) > 0 {
			if err := insertSnapshotTags(ctx, tx, record.ID, record.Tags); err != nil {
				return err
			}
		}

		return nil
	})

	return record.ID.String(), err
}

func (db *Database) ListSnapshots(ctx context.Context, example *DatabaseRecord, pageSize int) ([]*DatabaseRecord, error) {
	var queryBuilder strings.Builder
	var args []any
	var clauses []string

	// optional filters
	if example != nil {
		if example.ParentID.Valid {
			args = append(args, example.ParentID)
			clauses = append(clauses, fmt.Sprintf("hierarchy ~ ('*.' || $%d::text || '.*')::lquery", len(args)))
		}
		if example.TaskType.Valid && example.TaskType.String != "" {
			args = append(args, example.TaskType)
			clauses = append(clauses, fmt.Sprintf("task_type = $%d", len(args)))
		}
		if example.TaskName.Valid && example.TaskName.String != "" {
			args = append(args, example.TaskName)
			clauses = append(clauses, fmt.Sprintf("task_name = $%d", len(args)))
		}
		if example.TaskIndex.Valid {
			args = append(args, example.TaskIndex)
			clauses = append(clauses, fmt.Sprintf("task_index = $%d", len(args)))
		}
		if example.EnvType.Valid && example.EnvType.String != "" {
			args = append(args, example.EnvType)
			clauses = append(clauses, fmt.Sprintf("env_type = $%d", len(args)))
		}
		if example.SessionID.Valid {
			args = append(args, example.SessionID)
			clauses = append(clauses, fmt.Sprintf("session_id = $%d", len(args)))
		}
		if example.Step.Valid {
			args = append(args, example.Step)
			clauses = append(clauses, fmt.Sprintf("step = $%d", len(args)))
		}
		if example.ID != uuid.Nil {
			// use as page token for keyset pagination
			args = append(args, example.ID)
			clauses = append(clauses, fmt.Sprintf("id > $%d", len(args)))
		}
	}

	if example != nil && len(example.Tags) > 0 {
		tagsIdx := len(args) + 1
		args = append(args, example.Tags)
		clauses = append(clauses, fmt.Sprintf(`NOT EXISTS (
			SELECT 1 FROM unnest($%d::text[]) AS filter_tag
			WHERE NOT EXISTS (
				SELECT 1 FROM snapshot_tags st
				WHERE st.snapshot_id = snapshots.id AND st.tag = filter_tag
			)
		)`, tagsIdx))
	}

	clauses = append(clauses, "size IS NOT NULL") // only return completed snapshots

	// build query
	queryBuilder.WriteString(`
		SELECT id, parent_id, hierarchy, task_type, task_name, task_index, env_type, session_id, step, node, size, created_at
		FROM snapshots WHERE
	`)
	queryBuilder.WriteString(strings.Join(clauses, " AND "))

	// pagination
	args = append(args, pageSize)
	queryBuilder.WriteString(fmt.Sprintf(" ORDER BY id DESC LIMIT $%d", len(args)))

	stmt := queryBuilder.String()
	rows, err := db.conn.Query(ctx, stmt, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []*DatabaseRecord
	for rows.Next() {
		r := DatabaseRecord{}
		if err = rows.Scan(
			&r.ID,
			&r.ParentID,
			&r.Hierarchy,
			&r.TaskType,
			&r.TaskName,
			&r.TaskIndex,
			&r.EnvType,
			&r.SessionID,
			&r.Step,
			&r.Node,
			&r.Size,
			&r.CreatedAt,
		); err != nil {
			return nil, err
		}
		records = append(records, &r)
	}
	if err = rows.Err(); err != nil {
		return nil, err
	}

	if err := db.populateTags(ctx, records); err != nil {
		return nil, err
	}

	return records, nil
}

func (db *Database) GetSnapshot(ctx context.Context, id string) (*DatabaseRecord, error) {
	record := &DatabaseRecord{}

	if err := db.conn.QueryRow(ctx, `
		SELECT id, parent_id, hierarchy, task_type, task_name, task_index, env_type, session_id, step, node, size, created_at
		FROM snapshots WHERE id = $1 AND size IS NOT NULL
	`, id).Scan(
		&record.ID,
		&record.ParentID,
		&record.Hierarchy,
		&record.TaskType,
		&record.TaskName,
		&record.TaskIndex,
		&record.EnvType,
		&record.SessionID,
		&record.Step,
		&record.Node,
		&record.Size,
		&record.CreatedAt,
	); err != nil {
		return nil, err
	}

	tags, err := db.fetchTags(ctx, []uuid.UUID{record.ID})
	if err != nil {
		return nil, err
	}
	if len(tags) > 0 {
		record.Tags = tags[record.ID]
	}

	return record, nil
}

func (db *Database) SetSnapshotSize(ctx context.Context, id string, size uint64) error {
	sqlSize := sql.NullInt64{
		Int64: int64(size),
		Valid: true,
	}

	return db.withTx(ctx, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `UPDATE snapshots SET size = $2 WHERE id = $1`, id, sqlSize)
		return err
	})
}

func (db *Database) DeleteSnapshot(ctx context.Context, id string) error {
	return db.withTx(ctx, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `DELETE FROM snapshots WHERE id = $1`, id)

		// ignore if record does not exist
		if err != nil && !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		return nil
	})
}

func (db *Database) withTx(ctx context.Context, fn func(tx pgx.Tx) error) error {
	tx, err := db.conn.Begin(ctx)
	if err != nil {
		return err
	}

	if err = fn(tx); err != nil {
		_ = tx.Rollback(ctx)
		return err
	}

	return tx.Commit(ctx)
}

func (db *Database) migrate() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	db.logger.Info("running database migrations")

	return db.withTx(ctx, func(tx pgx.Tx) error {
		// try to acquire advisory lock to prevent concurrent migrations
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock($1)`, migrationAdvisoryLockID); err != nil {
			return err
		}

		for _, stmt := range migration {
			if _, err := tx.Exec(ctx, stmt); err != nil {
				return err
			}
		}

		// clear expired pending creations
		_, err := tx.Exec(ctx, `DELETE FROM snapshots WHERE size IS NULL AND created_at < now() - interval '10 minutes'`)

		return err
	})
}

func insertSnapshotTags(ctx context.Context, tx pgx.Tx, id uuid.UUID, tags []string) error {
	if len(tags) == 0 {
		return nil
	}

	batch := &pgx.Batch{}
	unique := make(map[string]struct{}, len(tags))
	for _, tag := range tags {
		tag = strings.TrimSpace(tag)
		if tag == "" {
			continue
		}
		if _, ok := unique[tag]; ok {
			continue
		}
		unique[tag] = struct{}{}
		batch.Queue(`INSERT INTO snapshot_tags (snapshot_id, tag) VALUES ($1, $2) ON CONFLICT (snapshot_id, tag) DO NOTHING`, id, tag)
	}
	if len(unique) == 0 {
		return nil
	}
	results := tx.SendBatch(ctx, batch)
	for range unique {
		if _, err := results.Exec(); err != nil && !errors.Is(err, pgx.ErrNoRows) {
			_ = results.Close()
			return err
		}
	}
	return results.Close()
}

func (db *Database) AddSnapshotTags(ctx context.Context, id string, tags []string) error {
	uid, err := uuid.Parse(id)
	if err != nil {
		return fmt.Errorf("invalid id: %w", err)
	}
	return db.withTx(ctx, func(tx pgx.Tx) error {
		return insertSnapshotTags(ctx, tx, uid, tags)
	})
}

func (db *Database) RemoveSnapshotTags(ctx context.Context, id string, tags []string) error {
	if len(tags) == 0 {
		return nil
	}
	return db.withTx(ctx, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `DELETE FROM snapshot_tags WHERE snapshot_id = $1::uuid AND tag = ANY($2::text[])`, id, tags)
		return err
	})
}

func (db *Database) SetSnapshotTags(ctx context.Context, id string, tags []string) error {
	uid, err := uuid.Parse(id)
	if err != nil {
		return fmt.Errorf("invalid id: %w", err)
	}
	return db.withTx(ctx, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `DELETE FROM snapshot_tags WHERE snapshot_id = $1::uuid`, uid); err != nil {
			return err
		}
		return insertSnapshotTags(ctx, tx, uid, tags)
	})
}

func (db *Database) fetchTags(ctx context.Context, ids []uuid.UUID) (map[uuid.UUID][]string, error) {
	if len(ids) == 0 {
		return map[uuid.UUID][]string{}, nil
	}

	rows, err := db.conn.Query(ctx, `SELECT snapshot_id, tag FROM snapshot_tags WHERE snapshot_id = ANY($1::uuid[]) ORDER BY tag`, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[uuid.UUID][]string, len(ids))
	for rows.Next() {
		var snapshotID uuid.UUID
		var tag string
		if err := rows.Scan(&snapshotID, &tag); err != nil {
			return nil, err
		}
		result[snapshotID] = append(result[snapshotID], tag)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	return result, nil
}

func (db *Database) populateTags(ctx context.Context, records []*DatabaseRecord) error {
	if len(records) == 0 {
		return nil
	}

	ids := make([]uuid.UUID, 0, len(records))
	index := make(map[uuid.UUID]*DatabaseRecord, len(records))
	for _, record := range records {
		ids = append(ids, record.ID)
		index[record.ID] = record
	}

	tags, err := db.fetchTags(ctx, ids)
	if err != nil {
		return err
	}

	for id, list := range tags {
		if record, ok := index[id]; ok {
			record.Tags = append([]string(nil), list...)
		}
	}
	return nil
}

func (db *Database) Close() {
	if db.conn != nil {
		db.conn.Close()
	}
}
