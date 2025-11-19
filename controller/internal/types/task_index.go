package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
)

// TaskIndex can be either string or int
type TaskIndex struct {
	Value interface{}
}

func (s TaskIndex) IsString() bool {
	_, ok := s.Value.(string)
	return ok
}

func (s TaskIndex) String() string {
	switch v := s.Value.(type) {
	case int:
		return fmt.Sprintf("%d", v)
	case string:
		return v
	default:
		return ""
	}
}

func (s TaskIndex) IsInt() bool {
	_, ok := s.Value.(int)
	return ok
}

func (s TaskIndex) Int() (int, error) {
	switch v := s.Value.(type) {
	case int:
		return v, nil
	case string:
		intValue, err := strconv.Atoi(v)
		if err != nil {
			return 0, fmt.Errorf("invalid TaskIndex value: %s", v)
		}
		return intValue, nil
	default:
		return 0, fmt.Errorf("invalid TaskIndex type")
	}
}

func (s TaskIndex) IsCustom() bool {
	val, _ := s.Int()
	return val == -1
}

func (s TaskIndex) Equals(other TaskIndex) bool {
	return reflect.DeepEqual(s, other)
}

func (s *TaskIndex) UnmarshalJSON(data []byte) error {
	var intValue int
	if err := json.Unmarshal(data, &intValue); err == nil {
		s.Value = intValue
		return nil
	}

	var stringValue string
	if err := json.Unmarshal(data, &stringValue); err == nil {
		s.Value = stringValue
		return nil
	}

	return fmt.Errorf("invalid value for TaskIndex")
}

func (s TaskIndex) MarshalJSON() ([]byte, error) {
	switch v := s.Value.(type) {
	case int:
		return json.Marshal(v)
	case string:
		return json.Marshal(v)
	default:
		return nil, fmt.Errorf("invalid value for TaskIndex")
	}
}

type NullTaskIndex struct {
	TaskIndex TaskIndex
	Valid     bool
}

func (s NullTaskIndex) String() string {
	if !s.Valid {
		return "NULL"
	}

	return s.TaskIndex.String()
}

func (s *NullTaskIndex) Scan(value any) error {
	if value == nil {
		s.TaskIndex.Value = nil
		s.Valid = false
		return nil
	}

	switch v := value.(type) {
	case int64:
		s.TaskIndex = TaskIndex{Value: int(v)}
		s.Valid = true
		return nil
	case string:
		// try to parse as int
		if intValue, err := strconv.Atoi(v); err == nil {
			s.TaskIndex = TaskIndex{Value: intValue}
		} else {
			s.TaskIndex = TaskIndex{Value: v}
		}
		s.Valid = true
		return nil
	}

	return fmt.Errorf("cannot scan %T into TaskIndex", value)
}

func (s NullTaskIndex) Value() (driver.Value, error) {
	if s.Valid {
		return s.String(), nil
	}

	return nil, nil
}
