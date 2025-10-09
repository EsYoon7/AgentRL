import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.timings = {}
        self.step_start_time = 0

    def step_start(self):
        self.timings = {}
        self.step_start_time = time.time()

    def step_end(self):
        self.timings["step"] = time.time() - self.step_start_time
        return self.timings

    @contextmanager
    def time(self, name):
        start = time.time()
        yield
        duration = time.time() - start
        self.timings[name] = duration
