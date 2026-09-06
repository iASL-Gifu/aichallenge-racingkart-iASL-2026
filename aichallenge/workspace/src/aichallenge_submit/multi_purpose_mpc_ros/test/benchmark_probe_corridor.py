"""Measure the actual controller snapshot scope on both configured paths.

Run from the package directory: python -m test.benchmark_probe_corridor
Uses no ROS node or vehicle commands. Timings exclude map loading and MPC.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import numpy as np

from .probe_support import configured_mpc
from .test_overtake_session import controller_method


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'multi_purpose_mpc_ros/mpc_controller.py'
    report = dict(python=platform.python_version(),
                  controller_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='context entry/copy and exit; excludes MPC, map load and ROS',
                  iterations=500, paths=[])
    probe = controller_method('_probe_corridor')
    for kind in ('center', 'race'):
        mpc = configured_mpc(kind)
        path = mpc.model.reference_path
        for populated in (False, True):
            if populated:
                # Populate real live diagnostic arrays before measuring their copies.
                path.target_lane_idx = 0 if kind == 'center' else None
                path.is_overtaking = path.target_lane_idx is not None
                wp = path.get_waypoint(30)
                mpc.model.update_states(wp.x, wp.y, wp.psi)
                with redirect_stdout(io.StringIO()):
                    mpc.get_control()
            samples = []
            for iteration in range(510):
                started = time.perf_counter_ns()
                with probe(None, mpc.model, 0 if kind == 'center' else None):
                    pass
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                if iteration >= 10:
                    samples.append(elapsed_ms)
            report['paths'].append(dict(
                path=kind, populated=populated, waypoints=path.n_waypoints,
                median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                max_ms=max(samples)))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
