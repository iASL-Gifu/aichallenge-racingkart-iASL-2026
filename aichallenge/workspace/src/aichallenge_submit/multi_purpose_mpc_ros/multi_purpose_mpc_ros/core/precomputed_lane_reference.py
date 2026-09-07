"""Immutable, objective-only lane references. No lane or solver state changes."""
import hashlib
import json
from pathlib import Path
import numpy as np


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_parameters(config):
    """The same design envelope for offline generation and startup validation."""
    def value(section, key):
        section = config[section] if isinstance(config, dict) else getattr(config, section)
        return float(section[key] if isinstance(section, dict) else getattr(section, key))
    return {f'{section}.{key}': value(section, key) for section, keys in (
        ('bicycle_model', ('length', 'width')),
        ('collision_geometry', ('length', 'width', 'rear_axle_to_center')),
        ('mpc', ('delta_max_deg', 'steer_rate_max', 'steering_tire_angle_gain_var',
                 'v_max', 'understeer_coeff')),
    ) for key in keys}


class PrecomputedLaneReference:
    def __init__(self, table, heading_gain=0.25):
        if not np.isfinite(heading_gain) or not 0.0 <= heading_gain <= 1.0:
            raise ValueError('precomputed L0 heading gain must be in [0, 1]')
        self.heading_gain = float(heading_gain)
        self.table = np.array(table, dtype=float, copy=True)
        self.table.setflags(write=False)

    def sample(self, wp_id, lane_idx):
        if lane_idx != 0:
            return None
        row = self.table[int(wp_id) % len(self.table)]
        return row if row[3] > 0.0 else None

    @classmethod
    def load(cls, csv_path, reference_path, package_root, parameters, heading_gain=0.25):
        path = Path(csv_path)
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['version'] != 1 or meta['parameters'] != parameters:
            raise ValueError('precomputed L0 design parameters changed; regenerate the profile')
        if file_digest(path) != meta['profile_sha256']:
            raise ValueError('precomputed L0 data checksum mismatch')
        for relative, expected in meta['sources'].items():
            if file_digest(Path(package_root) / relative) != expected:
                raise ValueError(f'precomputed L0 source changed: {relative}')
        rows = np.genfromtxt(path, delimiter=',', names=True)
        count = reference_path.n_waypoints
        ids = rows['wp_id'].astype(int)
        if (len(rows) < 4 or not np.array_equal(ids, rows['wp_id'])
                or np.any(np.diff(ids) != 1) or ids[0] < 0 or ids[-1] >= count):
            raise ValueError('invalid precomputed L0 waypoint indices')
        values = np.column_stack([rows[n] for n in ('e_y', 'e_psi', 'kappa', 'weight')])
        if (not np.isfinite(values).all() or np.any(values[:, 3] < 0)
                or np.any(values[:, 3] > 1) or values[0, 3] != 0 or values[-1, 3] != 0):
            raise ValueError('invalid precomputed L0 reference values or seams')
        for i, row in enumerate(rows):
            wp = reference_path.get_waypoint(int(ids[i]))
            if not np.allclose([wp.x, wp.y, wp.psi, wp.lb, wp.ub],
                               [row[n] for n in ('base_x', 'base_y', 'base_psi', 'base_lb', 'base_ub')],
                               atol=1e-6, rtol=0):
                raise ValueError(f'precomputed L0 geometry mismatch at WP{ids[i]}')
            upper, lower = reference_path.get_lane_bounds(int(ids[i]))[0]
            if not lower <= row['e_y'] <= upper:
                raise ValueError(f'precomputed L0 outside current lane at WP{ids[i]}')
        table = np.zeros((count, 4))
        table[ids] = values
        return cls(table, heading_gain)


def apply_precomputed_heading(reference_path, wp_id, lateral_targets, headings,
                              target_lane, soft_lane, has_soft_targets,
                              transition_weights):
    """Align the FINAL lateral objective, including Hybrid, with its heading.

    Return without touching full-width, Race return, L1/L2 or unrelated soft
    references. Do not modify dynamics, bounds, feedforward reservation or Q/R.
    Heading is tapered at the static profile seams to the existing objective.
    A limited gain retains the existing Center-heading objective: full geometric
    heading with the current high yaw cost and QP input-step limit can cause
    slow OSQP convergence. This is guidance, not exact curve tracking.
    """
    profile = getattr(reference_path, 'precomputed_lane_reference', None)
    if profile is None:
        return
    hybrid = transition_weights is not None and has_soft_targets and soft_lane == 0
    hard_l0 = target_lane == 0 and not (
        transition_weights is not None and has_soft_targets)
    if not (hybrid or hard_l0
            or (target_lane is None and soft_lane == 0 and not has_soft_targets)):
        return
    count = len(lateral_targets)
    for n in range(count):
        row = profile.sample(wp_id + n, 0)
        if row is None:
            continue
        if hard_l0:
            heading = row[1]
        else:
            # Use final blended targets, never the completed L0 heading during
            # a partial transition or a previous-prediction continuity blend.
            a, b = (n, n + 1) if n + 1 < count else (n - 1, n)
            wp = reference_path.get_waypoint(wp_id + n)
            wa = reference_path.get_waypoint(wp_id + a)
            wb = reference_path.get_waypoint(wp_id + b)
            dx = wb.x - lateral_targets[b] * np.sin(wb.psi) - wa.x + lateral_targets[a] * np.sin(wa.psi)
            dy = wb.y + lateral_targets[b] * np.cos(wb.psi) - wa.y - lateral_targets[a] * np.cos(wa.psi)
            heading = (np.arctan2(dy, dx) - wp.psi + np.pi) % (2 * np.pi) - np.pi
        headings[n] = profile.heading_gain * row[3] * heading
