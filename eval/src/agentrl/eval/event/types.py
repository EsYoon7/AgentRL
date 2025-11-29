from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypeAlias, TypeVar, Union, Sequence

from pydantic import BaseModel, Field

from ..session.types import MetricResult, MetricType, RunResult, RunSpec
from ..store.types import MetricsListItem, ResultListItem


class InitializedEvent(BaseModel):
    type: Literal['initialized'] = 'initialized'


class MetricsEvent(BaseModel):
    type: Literal['metrics'] = 'metrics'
    metric_type: MetricType
    items: Sequence[MetricsListItem]


class ResultListEvent(BaseModel):
    type: Literal['result_list'] = 'result_list'
    items: Sequence[ResultListItem]


class TokenUsageSummaryEvent(BaseModel):
    type: Literal['token_usage_summary'] = 'token_usage_summary'
    summary: str


class SessionStartedEvent(BaseModel):
    type: Literal['session_started'] = 'session_started'
    spec: RunSpec
    session_id: int


class SessionCompletedEvent(BaseModel):
    type: Literal['session_completed'] = 'session_completed'
    spec: RunSpec
    session_id: Optional[int]
    result: RunResult
    metric: tuple[Optional[float], MetricResult]


class SpecsEvent(BaseModel):
    type: Literal['specs'] = 'specs'
    items: Sequence[RunSpec]


class WorkflowStartEvent(BaseModel):
    type: Literal['workflow_start'] = 'workflow_start'
    path: Optional[Path]


class WorkflowCompletedEvent(BaseModel):
    type: Literal['workflow_completed'] = 'workflow_completed'


Event: TypeAlias = Annotated[
    Union[
        InitializedEvent,
        MetricsEvent,
        ResultListEvent,
        TokenUsageSummaryEvent,
        SessionStartedEvent,
        SessionCompletedEvent,
        SpecsEvent,
        WorkflowStartEvent,
        WorkflowCompletedEvent
    ],
    Field(discriminator='type')
]

T = TypeVar('T', bound=Event)

Listener = Callable[[T], Optional[Coroutine[Any, Any, None]]]
