from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class MetricsListItem(BaseModel):
    model: str
    task: str
    valid: int
    avg: float
    std: Optional[float]
    bon: Optional[float]


class ResultListItem(BaseModel):
    name: str
    path: Path
    ts: int = 0
