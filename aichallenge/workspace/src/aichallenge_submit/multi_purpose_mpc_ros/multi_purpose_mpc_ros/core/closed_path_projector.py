"""Prepared, exact-coordinate cached projection onto an immutable closed path."""
from collections import OrderedDict
import math

import numpy as np


class ClosedPathProjector:
    """Compare every segment, retaining the first segment on equal distance.

    Only geometry is cached: no traffic, safety decision, or MPC solution is
    retained. Coordinates are never rounded. Construct a new instance whenever
    the reference geometry changes. The LRU bounds memory even on long runs.
    """

    def __init__(self, points, cumulative, total_length, cache_size=2048):
        self.total_length = float(total_length)
        self.cache_size = max(0, int(cache_size))
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        xy = np.array(points, dtype=float, copy=True).reshape(-1, 2)
        delta = np.roll(xy, -1, axis=0) - xy
        length_sq = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
        valid = length_sq > 1e-12
        self.x0, self.y0 = xy[valid].T
        self.vx, self.vy = delta[valid].T
        self.length_sq = length_sq[valid]
        self.length = np.sqrt(self.length_sq)
        self.s0 = np.asarray(cumulative[:len(xy)], dtype=float)[valid] if len(xy) >= 2 else np.zeros(0)
        self._valid = len(xy) >= 2 and self.total_length > 0 and bool(np.any(valid))

    def project(self, x, y):
        key = (float(x), float(y))
        if not self._valid or not all(map(math.isfinite, key)):
            return None
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        px, py = key
        ratio = np.clip(((px - self.x0) * self.vx + (py - self.y0) * self.vy)
                        / self.length_sq, 0., 1.)
        dx = px - (self.x0 + ratio * self.vx)
        dy = py - (self.y0 + ratio * self.vy)
        distance_sq = dx * dx + dy * dy
        index = int(np.argmin(distance_sq))
        if not math.isfinite(float(distance_sq[index])):
            return None
        result = (float((self.s0[index] + ratio[index] * self.length[index]) % self.total_length),
                  float(dx[index] * (-self.vy[index] / self.length[index])
                        + dy[index] * (self.vx[index] / self.length[index])))
        if self.cache_size:
            self._cache[key] = result
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result
