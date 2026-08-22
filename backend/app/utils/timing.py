"""
StageTimer — lightweight context manager that records wall-clock time per stage.

Usage:
    timer = StageTimer()
    with timer.stage("embedding"):
        vector = model.encode(text)
    with timer.stage("retrieval"):
        results = qdrant.search(...)
    print(timer.all_ms())   # {"embedding": 12.3, "retrieval": 18.7, "total": 31.0}
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional


class StageTimer:
    def __init__(self) -> None:
        self._stages: Dict[str, float] = {}
        self._start_wall: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._stages[name] = round(elapsed_ms, 2)

    def record(self, name: str, ms: float) -> None:
        """Manually record a pre-measured duration."""
        self._stages[name] = round(ms, 2)

    def get(self, name: str) -> Optional[float]:
        return self._stages.get(name)

    def all_ms(self) -> Dict[str, float]:
        total = round((time.perf_counter() - self._start_wall) * 1000, 2)
        return {**self._stages, "total": total}

    def reset(self) -> None:
        self._stages.clear()
        self._start_wall = time.perf_counter()
