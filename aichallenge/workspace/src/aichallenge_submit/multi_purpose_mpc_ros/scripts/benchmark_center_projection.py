#!/usr/bin/env python3
"""Short, ROS-free projection benchmark; not an end-to-end control benchmark."""
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from multi_purpose_mpc_ros.core.closed_path_projector import ClosedPathProjector
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    build_closed_path_arc_lengths, project_to_closed_path_frenet,
)


def main():
    with (PACKAGE / 'env/centerline/traj_center_mincurv_capped.csv').open() as stream:
        points = [(float(r['x_m']), float(r['y_m'])) for r in csv.DictReader(stream)]
    geometry = build_closed_path_arc_lengths(points)
    rng = np.random.default_rng(260)
    queries = np.asarray(points) + rng.uniform(-4., 4., (len(points), 2))
    prepared = ClosedPathProjector(*geometry, cache_size=0)
    cached = ClosedPathProjector(*geometry)
    legacy = lambda x, y: project_to_closed_path_frenet(x, y, *geometry)
    maximum_error = max(max(abs(a-b) for a, b in zip(legacy(*q), prepared.project(*q)))
                        for q in queries)
    rows = {}
    for name, project in [('scalar', legacy), ('prepared_no_cache', prepared.project),
                          ('exact_cache_hit', cached.project)]:
        for q in queries:
            project(*q)
        times = []
        for _ in range(3):
            start = perf_counter()
            for q in queries:
                project(*q)
            times.append((perf_counter() - start) / len(queries) * 1e6)
        rows[name + '_us_per_projection'] = float(np.median(times))
    print(json.dumps(dict(points=len(points), queries=len(queries),
                          max_absolute_error=maximum_error, **rows), indent=2))


if __name__ == '__main__':
    main()
