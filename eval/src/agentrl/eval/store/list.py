from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .store import ResultStore
from .types import ResultListItem


class ResultList:

    def __init__(self,
                 base_dir: Path,
                 *,
                 model: Optional[str] = None,
                 task: Optional[str] = None,
                 view_only: bool = False):
        """
        :param model: for evaluation that tests multiple models, pass None
        :param task: for evaluation that tests multiple tasks, pass None
        :param view_only: in view-only mode, if no model or task is specified, all results are listed
        """
        self.base_dir = base_dir
        self.multi_model = model is None
        self.multi_task = task is None

        if not view_only:
            if model is None:
                model = 'multi-model'
            if task is None:
                task = 'multi-task'
        if task is None:
            if model is None:
                self.prefix = ''
            else:
                self.prefix = f'{model}-'
        else:
            if model is None:
                model = 'multi-model'
            self.prefix = f'{model}-{task}-'

    def list(self) -> list[ResultListItem]:
        if not self.base_dir.exists():
            return []

        if not self.base_dir.is_dir():
            raise NotADirectoryError(f'"{self.base_dir}" is not a directory.')

        result: list[ResultListItem] = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith(self.prefix):
                # try to parse date for sorting
                match = re.search(r'(\d{12})$', item.name)
                ts = 0
                if match:
                    try:
                        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
                        ts = int(dt.timestamp())
                    except ValueError:
                        pass
                result.append(ResultListItem(name=item.name, path=item, ts=ts))

        return sorted(result, key=lambda x: x.ts, reverse=True)

    def get(self, name: str) -> ResultStore:
        store_path = self.base_dir / name
        return ResultStore(path=store_path, multi_model=self.multi_model, multi_task=self.multi_task)

    def create(self) -> ResultStore:
        store_path = self.base_dir / (self.prefix + datetime.now().strftime('%Y%m%d%H%M'))
        return ResultStore(path=store_path, multi_model=self.multi_model, multi_task=self.multi_task)

    def path(self, path: Path, *, resume: bool = False) -> ResultStore:
        return ResultStore(path=path, resume=resume, multi_model=self.multi_model, multi_task=self.multi_task)
