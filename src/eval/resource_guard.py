from __future__ import annotations

"""
Safety watchdog for stress testing on a single laptop with no memory headroom to spare.

Runs in a background thread, polling available system memory at a fixed interval. If available
memory drops below a floor, it sets an internal flag (`should_abort`) that stress tests check
between steps/iterations and stop on. This is COOPERATIVE, not a hard kill -- a hard kill
mid-inference can leave more memory pinned (e.g. a model half-loaded) than a clean, checked stop
does, so every stress test in this package is written to check `guard.should_abort` at safe
boundaries (between ramp steps, between calls) rather than mid-computation.

Also records peak memory used and the lowest available memory seen during a run, so each
report answers "how close did this get to the edge", not just "did it crash."
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass
class ResourceGuard:
    min_free_mb: int = 1536          # hard floor: signal abort if available memory drops below this
    poll_interval_s: float = 1.0
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _abort_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    peak_used_mb: float = field(default=0.0, init=False)
    min_available_seen_mb: float = field(default=float("inf"), init=False)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval_s * 2)

    @property
    def should_abort(self) -> bool:
        return self._abort_event.is_set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            used_mb = vm.used / (1024 * 1024)
            self.peak_used_mb = max(self.peak_used_mb, used_mb)
            self.min_available_seen_mb = min(self.min_available_seen_mb, available_mb)
            if available_mb < self.min_free_mb and not self._abort_event.is_set():
                self._abort_event.set()
                print(
                    f"[resource_guard] ABORT signal: available memory {available_mb:.0f}MB "
                    f"< floor {self.min_free_mb}MB. Stress test will stop at its next checkpoint."
                )
            time.sleep(self.poll_interval_s)

    def snapshot(self) -> dict:
        vm = psutil.virtual_memory()
        return {
            "available_mb": round(vm.available / (1024 * 1024), 1),
            "used_mb": round(vm.used / (1024 * 1024), 1),
            "percent_used": vm.percent,
        }
