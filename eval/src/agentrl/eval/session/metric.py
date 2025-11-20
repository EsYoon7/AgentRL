from __future__ import annotations

from typing import Optional

from .types import MetricResult, MetricType, RunResult, SampleStatus


class Metric:

    def __init__(self, metric_type: MetricType):
        self.type = metric_type

    def __call__(self, result: RunResult) -> tuple[Optional[float], MetricResult]:
        if self.type == MetricType.REWARD:
            value = result.reward
        elif self.type == MetricType.SCORE:
            value = result.score
        elif self.type == MetricType.SUCCESS_RATE:
            value = 1.0 if result.reward == 1.0 else 0.0 if result.reward is not None else None
        else:
            raise ValueError(f'unsupported metric type: {self.type}')

        if not result.status.is_completed():
            return value, MetricResult.ERROR
        if value == 1.0:
            return value, MetricResult.SUCCESS
        if value == 0.0:
            return value, MetricResult.FAILURE
        if isinstance(value, (int, float)) and 0.0 < value < 1.0:
            return value, MetricResult.PARTIAL_SUCCESS
        return value, MetricResult.UNKNOWN

    @staticmethod
    def all(result: RunResult) -> dict[MetricType, Optional[float]]:
        return {
            metric: Metric(metric)(result)[0]
            for metric in MetricType
        }
