"""Independent per-job memory watchdog around an existing worker controller."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import time

GIB = 1024 ** 3
INTERVAL_SEC = 0.5


def read_memory():
    values = {}
    with open('/proc/meminfo', encoding='ascii') as stream:
        for line in stream:
            key, _, value = line.partition(':')
            if key in ('MemAvailable', 'SwapTotal', 'SwapFree'):
                values[key] = int(value.split()[0]) * 1024
    return {'available_bytes': values['MemAvailable'],
            'swap_used_bytes': values['SwapTotal'] - values['SwapFree']}


class MemoryGuardedController:
    """Preserve controller cancellation/cleanup. Add a watchdog and an audit log."""
    def __init__(self, factory, *, on_finished, audit_path, probe=read_memory,
                 interval_sec=INTERVAL_SEC):
        self._lock = threading.RLock()
        self._active = None
        self._probe = probe
        self._interval = interval_sec
        self._audit_path = Path(audit_path)
        self._on_finished = on_finished
        self._controller = factory(on_finished=self._finished)

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def _audit(self, event, state, **values):
        # A log failure must never prevent terminating a memory-heavy worker.
        row = {'event': event, 'wall_time': time.time(), 'job_id': state.get('job_id'),
               'min_available_bytes': GIB, 'max_job_swap_growth_bytes': GIB,
               'sample_interval_sec': self._interval, **values}
        try:
            with self._audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        except OSError:
            pass

    def start(self, *args, **kwargs):
        return self._start('start', *args, **kwargs)

    def start_warm(self, *args, **kwargs):
        return self._start('start_warm', *args, **kwargs)

    def _start(self, method, *args, **kwargs):
        from .controller import BusyError
        baseline = self._probe()  # A broken probe rejects dispatch before spawning.
        state = {'done': threading.Event(), 'baseline': baseline, 'tripped': None,
                 'job_id': None, 'min_observed_available_bytes': baseline['available_bytes'],
                 'max_observed_swap_growth_bytes': 0}
        original_started = kwargs.pop('on_started', None)
        with self._lock:
            if self._active is not None:
                raise BusyError('a guarded reconstruction worker is already active')
            self._active = state
        def started(job_id):
            state['job_id'] = job_id
            if original_started is not None:
                original_started(job_id)
            self._audit('memory_guard_started', state, baseline=baseline)
            threading.Thread(target=self._watch, args=(state,), daemon=True).start()
        try:
            return getattr(self._controller, method)(*args, on_started=started, **kwargs)
        except BaseException:
            state['done'].set()
            with self._lock:
                if self._active is state:
                    self._active = None
            raise

    def _watch(self, state):
        while not state['done'].is_set():
            try:
                sample = self._probe()
                growth = sample['swap_used_bytes'] - state['baseline']['swap_used_bytes']
                reason = ('available_memory_below_1_gib' if sample['available_bytes'] < GIB
                          else 'job_swap_growth_above_1_gib' if growth > GIB else '')
            except Exception as error:
                sample, growth = {}, 0
                reason = 'memory_probe_failed:' + type(error).__name__
            with self._lock:
                if self._active is not state or state['done'].is_set():
                    return
                state['min_observed_available_bytes'] = min(
                    state['min_observed_available_bytes'], sample.get('available_bytes', GIB))
                state['max_observed_swap_growth_bytes'] = max(state['max_observed_swap_growth_bytes'], growth)
                if reason:
                    state['tripped'] = reason
                    # Holding this lock prevents a stale sample from cancelling a new job.
                    accepted = self._controller.cancel()
                    if not accepted:
                        state['tripped'] = None
                    self._audit('memory_guard_triggered', state, reason=reason, sample=sample,
                                job_swap_growth_bytes=growth, cancel_accepted=accepted)
                    return
            state['done'].wait(self._interval)

    def _finished(self, finished):
        with self._lock:
            state = self._active
            if state is not None:
                state['done'].set()
                self._active = None
                if state['tripped'] and finished.cause != 'cleanup_failed':
                    finished = replace(finished, cause='resource_guard')
                self._audit('memory_guard_finished', state, cause=finished.cause,
                            returncode=finished.returncode, elapsed_sec=finished.elapsed_sec,
                            trigger=state['tripped'],
                            min_observed_available_bytes=state['min_observed_available_bytes'],
                            max_observed_swap_growth_bytes=state['max_observed_swap_growth_bytes'])
        self._on_finished(finished)
