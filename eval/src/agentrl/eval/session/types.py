from __future__ import annotations

from datetime import datetime
from enum import auto, Enum
from typing import Any, Optional, Sequence, TypeAlias, Union

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    # python 3.10 fallback
    class StrEnum(str, Enum):
        pass

from pydantic import BaseModel, model_validator

TaskIndex: TypeAlias = Union[int, str]


class SampleStatus(StrEnum):
    # status values returned by the controller API
    UNKNOWN = 'unknown'
    RUNNING = 'running'
    COMPLETED = 'completed'
    VALIDATION_FAILED = 'agent validation failed'
    INVALID_ACTION = 'agent invalid action'
    TASK_LIMIT_REACHED = 'task limit reached'
    TASK_ERROR = 'task error'
    CANCELLED = 'cancelled'

    # status values defined by ourselves
    MODEL_ERROR = 'model error'
    SERVER_ERROR = 'server error'
    WORKFLOW_ERROR = 'workflow error'

    def is_completed(self) -> bool:
        return self in self.completed_statuses()

    @staticmethod
    def completed_statuses() -> set[SampleStatus]:
        return {
            SampleStatus.UNKNOWN,
            SampleStatus.COMPLETED,
            SampleStatus.VALIDATION_FAILED,
            SampleStatus.INVALID_ACTION,
            SampleStatus.TASK_LIMIT_REACHED
        }

    def is_client_error(self) -> bool:
        return self in self.client_errors()

    @staticmethod
    def client_errors() -> set[SampleStatus]:
        return {
            SampleStatus.CANCELLED,
            SampleStatus.MODEL_ERROR,
            SampleStatus.SERVER_ERROR,
            SampleStatus.WORKFLOW_ERROR,
        }


class InteractResponse(BaseModel):
    """
    Note: based on different versions of the controller API,
          not all fields may be present.
    """
    finish: Optional[bool] = None
    reward: Optional[float] = None
    status: Optional[SampleStatus] = None
    messages: Optional[Sequence[dict]] = None
    tools: Optional[Sequence[dict]] = None
    metrics: Optional[dict] = None
    result: Optional[Any] = None

    @model_validator(mode='after')
    def messages_not_empty(self):
        if self.finish:
            if not self.status:
                # enforce default status when finished
                self.status = SampleStatus.UNKNOWN

        else:
            if not self.messages:
                raise ValueError('interact response messages should not be empty')

        return self


class RunSpec(BaseModel):
    model: str
    run: int
    task: str
    index: TaskIndex
    custom_params: Optional[dict[str, Any]]

    def task_key(self) -> str:
        return f'{self.task}:{self.index}'

    def run_key(self) -> str:
        return f'{self.model}:{self.task}:{self.index}:{self.run}'


class RunResult(BaseModel):
    model: str
    session_id: Optional[int]
    run: int
    task: str
    index: TaskIndex
    status: SampleStatus
    reward: Optional[float]
    score: Optional[float]
    metrics: Optional[dict[str, Any]]
    result: Optional[Any]
    task_trace: Optional[Any]
    raw_trace: Optional[Any]
    time_start: Optional[datetime]
    time_end: Optional[datetime]

    def spec(self) -> RunSpec:
        return RunSpec(
            model=self.model,
            run=self.run,
            task=self.task,
            index=self.index,
            custom_params=None
        )


class MetricType(StrEnum):
    REWARD = 'reward'
    SCORE = 'score'
    SUCCESS_RATE = 'success_rate'


class MetricResult(Enum):
    SUCCESS = auto()
    PARTIAL_SUCCESS = auto()
    FAILURE = auto()
    UNKNOWN = auto()
    ERROR = auto()
