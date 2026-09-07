"""Bounded wall/thread CPU measurements. No subprocesses in the control loop."""
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import time
import threading
from functools import wraps


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {'n': len(values), 'p50': round(float(values[len(values)//2]), 3),
            'p95': round(float(values[min(len(values)-1, int(.95*len(values)))]), 3),
            'max': round(float(values[-1]), 3)}


class RuntimeDiagnostics:
    def __init__(self, period_sec, interval_sec=5.0):
        self.period = float(period_sec)
        self.interval = max(float(interval_sec), 1.0)
        self.last_report = time.perf_counter()
        self.last_process_cpu = time.process_time()
        self.last_start = None
        self.last_sim = None
        self.last_publish = None
        self.samples = defaultdict(lambda: deque(maxlen=1024))
        self.counts = Counter()
        self.active = False
        self.details = {}
        self.owner_thread = None
        self.wait_ms = 0.0
        self.mpc_ms = 0.0

    def begin(self, sim_sec):
        self.start = time.perf_counter()
        self.thread_start = time.thread_time()
        if self.last_start is not None:
            wall_dt = self.start-self.last_start
            self.samples['cycle_wall_ms'].append(1000*wall_dt)
            if wall_dt > 0 and sim_sec >= self.last_sim:
                self.samples['sim_wall_ratio'].append((sim_sec-self.last_sim)/wall_dt)
            elif sim_sec < self.last_sim:
                self.counts['sim_clock_backwards'] += 1
        self.last_start, self.last_sim = self.start, sim_sec
        self.wait_ms = self.mpc_ms = 0.0
        self.active = True
        self.owner_thread = threading.get_ident()
        self.stage_start = self.stage_cpu_start = None

    def start_stages(self):
        self.stage_start = time.perf_counter()
        self.stage_cpu_start = time.thread_time()

    def checkpoint(self, name):
        if not self.active or self.stage_start is None:
            return
        now, cpu = time.perf_counter(), time.thread_time()
        self.samples['stage_'+name+'_wall_ms'].append((now-self.stage_start)*1000)
        self.samples['stage_'+name+'_thread_cpu_ms'].append((cpu-self.stage_cpu_start)*1000)
        self.stage_start, self.stage_cpu_start = now, cpu

    def recording_detail(self):
        return self.active and self.owner_thread == threading.get_ident()

    def record_detail(self, name, wall_ms, cpu_ms, error=False):
        if not self.recording_detail():
            return
        row = self.details.setdefault(name, dict(calls=0, wall_total_ms=0.,
            cpu_total_ms=0., wall_max_ms=0., cpu_max_ms=0., exceptions=0))
        row['calls'] += 1
        row['wall_total_ms'] += wall_ms
        row['cpu_total_ms'] += cpu_ms
        row['wall_max_ms'] = max(row['wall_max_ms'], wall_ms)
        row['cpu_max_ms'] = max(row['cpu_max_ms'], cpu_ms)
        row['exceptions'] += bool(error)

    def instrument(self, instance, name, label):
        original = getattr(instance, name)
        original = getattr(original, '_timing_original', original)

        @wraps(original)
        def measured(*args, **kwargs):
            if not self.recording_detail():
                return original(*args, **kwargs)
            metric = label() if callable(label) else label
            start, cpu = time.perf_counter(), time.thread_time()
            error = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                error = True
                raise
            finally:
                self.record_detail(metric, (time.perf_counter()-start)*1000,
                                   (time.thread_time()-cpu)*1000, error)
        measured._timing_original = original
        setattr(instance, name, measured)

    def record_mpc(self, role, wall_ms, cpu_ms, status, fallback, error=False):
        if not self.active:
            return
        self.mpc_ms += wall_ms
        self.samples[role+'_wall_ms'].append(wall_ms)
        self.samples[role+'_thread_cpu_ms'].append(cpu_ms)
        self.counts[role+'_calls'] += 1
        self.counts[role+':'+str(status)] += 1
        self.counts[role+'_prediction_fallback'] += bool(fallback)
        self.counts[role+'_exception'] += bool(error)

    def published(self):
        now = time.perf_counter()
        if self.last_publish is not None:
            self.samples['command_gap_wall_ms'].append(1000*(now-self.last_publish))
        self.last_publish = now
        self.counts['command_publishes'] += 1

    def finish(self, fallback=False):
        self.checkpoint("tail_or_early_return")
        wall_ms = (time.perf_counter()-self.start)*1000
        cpu_ms = (time.thread_time()-self.thread_start)*1000
        work_ms = max(0., wall_ms-self.wait_ms)
        self.samples['work_wall_ms'].append(work_ms)
        self.samples['work_thread_cpu_ms'].append(cpu_ms)
        self.samples['rate_wait_wall_ms'].append(self.wait_ms)
        self.samples['non_mpc_work_wall_ms'].append(max(0., work_ms-self.mpc_ms))
        # Includes scheduler/GIL/I/O/locks; not a pure CPU contention metric.
        self.samples['non_thread_wall_ms'].append(max(0., work_ms-cpu_ms))
        self.counts['cycles'] += 1
        self.counts['work_over_period'] += work_ms > 1000*self.period
        self.counts['fallback_owned_cycles'] += bool(fallback)
        self.active = False

    def report(self, context):
        now = time.perf_counter()
        elapsed = now-self.last_report
        if elapsed < self.interval:
            return None
        cpu = time.process_time()
        result = dict(window_wall_sec=round(elapsed, 3), pid=os.getpid(),
                      period_ms=1000*self.period,
                      process_cpu_pct=round(100*(cpu-self.last_process_cpu)/elapsed, 1),
                      context=context, counts=dict(self.counts),
                      detail_functions={name: {key: round(value, 3) if isinstance(value, float) else value
                                              for key, value in row.items()}
                                        for name, row in self.details.items()},
                      metrics={key: summary(values) for key, values in self.samples.items()})
        self.details.clear()
        self.samples.clear()
        self.counts.clear()
        self.last_report, self.last_process_cpu = now, cpu
        return '[ControlTiming] '+json.dumps(result, separators=(',', ':'), allow_nan=False)


def runtime_identity(paths):
    """Hash actual imported files/config once, not source-tree assumptions."""
    result = {'pid': os.getpid(), 'cpu_count': os.cpu_count(), 'files': {},
              'threads_env': {key: os.environ.get(key, '') for key in (
                  'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    for name, path in paths.items():
        try:
            path = Path(path).resolve()
            result['files'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        except (OSError, TypeError) as error:
            result['files'][name] = {'error': str(error)}
    return '[RuntimeIdentity] '+json.dumps(result, separators=(',', ':'))
