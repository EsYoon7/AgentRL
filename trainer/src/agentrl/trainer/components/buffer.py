import asyncio
from collections import deque
from math import lcm


class Buffer:
    def __init__(self, max_size, group_size):
        self.max_size = max_size
        self.strict_group = group_size > 1
        self.group_size = group_size
        self.groups = {}
        self.put_signal = asyncio.Event()
        self.get_signal = asyncio.Event()
        self.queue = deque()

    async def add(self, item):
        while len(self.queue) >= self.max_size:
            await self.get_signal.wait()
            self.get_signal.clear()
        if self.strict_group:
            group_id = item["group_id"]
            if group_id not in self.groups:
                self.groups[group_id] = []
            self.groups[group_id].append(item)
            if len(self.groups[group_id]) >= self.group_size:
                self.queue.extend(self.groups[group_id])
                self.put_signal.set()
                del self.groups[group_id]
        else:
            self.queue.append(item)
            self.put_signal.set()

    async def get(self, minimum, multiple: int = 1):
        if self.strict_group:
            multiple = lcm(self.group_size, multiple)
        while True:
            length = len(self.queue) // multiple * multiple
            if length >= minimum:
                items = [self.queue.popleft() for _ in range(length)]
                self.get_signal.set()
                return items
            await self.put_signal.wait()
            self.put_signal.clear()
