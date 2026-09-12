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
    def __init__(self, table, heading_gain=0.25, lane_idx=0):
        if lane_idx not in (0, 2):
            raise ValueError("precomputed lane must be L0 or L2")
        self.lane_idx = lane_idx
        if not np.isfinite(heading_gain) or not 0.0 <= heading_gain <= 1.0:
            raise ValueError('precomputed lane heading gain must be in [0, 1]')
        self.heading_gain = float(heading_gain)
        self.table = np.array(table, dtype=float, copy=True)
        self.table.setflags(write=False)

    def sample(self, wp_id, lane_idx):
        if lane_idx != self.lane_idx:
            return None
        row = self.table[int(wp_id) % len(self.table)]
        return row if row[3] > 0.0 else None

    @classmethod
    def load(cls, csv_path, reference_path, package_root, parameters, heading_gain=0.25):
        path = Path(csv_path)
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['version'] != 1 or meta['parameters'] != parameters:
            raise ValueError('precomputed lane design parameters changed; regenerate the profile')
        if file_digest(path) != meta['profile_sha256']:
            raise ValueError('precomputed lane data checksum mismatch')
        for relative, expected in meta['sources'].items():
            # ament installs Python code separately from share/env and config.
            # Verify the running implementation, not a nonexistent share copy.
            source = (Path(__file__).with_name('reference_path.py')
                      if relative == 'multi_purpose_mpc_ros/core/reference_path.py'
                      else Path(package_root) / relative)
            if file_digest(source) != expected:
                raise ValueError(f'precomputed lane source changed: {relative}')
        lane_idx = meta.get('lane_idx', 0)
        if lane_idx not in (0, 2):
            raise ValueError('invalid precomputed lane index')
        rows = np.genfromtxt(path, delimiter=',', names=True)
        count = reference_path.n_waypoints
        ids = rows['wp_id'].astype(int)
        if (len(rows) < 4 or not np.array_equal(ids, rows['wp_id'])
                or np.any(np.diff(ids) != 1) or ids[0] < 0 or ids[-1] >= count):
            raise ValueError('invalid precomputed lane waypoint indices')
        values = np.column_stack([rows[n] for n in ('e_y', 'e_psi', 'kappa', 'weight')])
        if (not np.isfinite(values).all() or np.any(values[:, 3] < 0)
                or np.any(values[:, 3] > 1) or values[0, 3] != 0 or values[-1, 3] != 0):
            raise ValueError('invalid precomputed lane reference values or seams')
        for i, row in enumerate(rows):
            wp = reference_path.get_waypoint(int(ids[i]))
            if not np.allclose([wp.x, wp.y, wp.psi, wp.lb, wp.ub],
                               [row[n] for n in ('base_x', 'base_y', 'base_psi', 'base_lb', 'base_ub')],
                               atol=1e-6, rtol=0):
                raise ValueError(f'precomputed lane geometry mismatch at WP{ids[i]}')
            upper, lower = reference_path.get_lane_bounds(int(ids[i]))[lane_idx]
            if not lower <= row['e_y'] <= upper:
                raise ValueError(f'precomputed lane outside current lane at WP{ids[i]}')
        table = np.zeros((count, 4))
        table[ids] = values
        return cls(table, heading_gain, lane_idx)


def apply_precomputed_heading(reference_path, wp_id, lateral_targets, headings,
                              target_lane, soft_lane, has_soft_targets,
                              transition_weights):
    """Align the FINAL lateral objective, including Hybrid, with its heading.

    Return without touching full-width, Race return, L1 or unrelated soft
    references. Do not modify dynamics, bounds, feedforward reservation or Q/R.
    Heading is tapered at the static profile seams to the existing objective.
    The legacy profile uses a partial heading gain; extended outer profiles
    use their geometric heading with gain 1.0 inside the seam tapers.
    This is an objective, not a guarantee of closed-loop curve tracking.
    """
    profile = getattr(reference_path, 'precomputed_lane_reference', None)
    if profile is None:
        return
    lane = soft_lane if transition_weights is not None and has_soft_targets else target_lane
    if lane is None and not has_soft_targets:
        lane = soft_lane
    hybrid = transition_weights is not None and has_soft_targets and soft_lane in (0, 2)
    hard_outer = target_lane in (0, 2) and not (
        transition_weights is not None and has_soft_targets)
    if not (hybrid or hard_outer
            or (target_lane is None and soft_lane in (0, 2) and not has_soft_targets)):
        return
    count = len(lateral_targets)
    for n in range(count):
        row = profile.sample(wp_id + n, lane)
        if row is None:
            continue
        if hard_outer:
            heading = row[1]
        else:
            # Use final blended targets, never the completed outer-lane heading during
            # a partial transition or a previous-prediction continuity blend.
            a, b = (n, n + 1) if n + 1 < count else (n - 1, n)
            wp = reference_path.get_waypoint(wp_id + n)
            wa = reference_path.get_waypoint(wp_id + a)
            wb = reference_path.get_waypoint(wp_id + b)
            dx = wb.x - lateral_targets[b] * np.sin(wb.psi) - wa.x + lateral_targets[a] * np.sin(wa.psi)
            dy = wb.y + lateral_targets[b] * np.cos(wb.psi) - wa.y - lateral_targets[a] * np.cos(wa.psi)
            heading = (np.arctan2(dy, dx) - wp.psi + np.pi) % (2 * np.pi) - np.pi
        headings[n] = profile.heading_gain * row[3] * heading


class PrecomputedLaneReferences:
    """Profiles for both outer lanes sharing the existing Center basis."""
    def __init__(self, profiles):
        self.profiles = {p.lane_idx: p for p in profiles}
        if len(self.profiles) != len(profiles):
            raise ValueError('duplicate precomputed lane')
        gains = {p.heading_gain for p in profiles}
        if len(gains) != 1:
            raise ValueError('lane profiles must use the same heading gain')
        self.heading_gain = gains.pop()

    def sample(self, wp_id, lane_idx):
        profile = self.profiles.get(lane_idx)
        return None if profile is None else profile.sample(wp_id, lane_idx)


def apply_precomputed_curvature(reference_path, wp_id, lateral_targets,
                                curvature_targets, target_lane, soft_lane,
                                has_soft_targets, transition_weights):
    """Change only the input objective, never the Center dynamics or bounds.

    Hard lanes use the certified curve. Hybrid uses the same final blended
    lateral geometry as the heading objective, not the completed outer lane.
    """
    changed = np.zeros(len(curvature_targets), dtype=bool)
    profile = getattr(reference_path, 'precomputed_lane_reference', None)
    if profile is None:
        return changed
    hybrid = transition_weights is not None and has_soft_targets
    lane = soft_lane if hybrid else target_lane
    if lane is None and not has_soft_targets:
        lane = soft_lane
    if lane not in (0, 2) or (target_lane is None and has_soft_targets and not hybrid):
        return changed
    for n in range(len(curvature_targets)):
        row = profile.sample(wp_id+n, lane)
        if row is None:
            continue
        kappa = row[2]
        if hybrid:
            i = min(max(n, 1), len(lateral_targets)-2)
            points = []
            for j in (i-1, i, i+1):
                wp = reference_path.get_waypoint(wp_id+j)
                points.append((wp.x-lateral_targets[j]*np.sin(wp.psi),
                               wp.y+lateral_targets[j]*np.cos(wp.psi)))
            a, b = np.diff(np.asarray(points), axis=0)
            denominator = np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(a+b)
            if denominator <= 1e-9:
                continue
            kappa = 2*(a[0]*b[1]-a[1]*b[0])/denominator
        curvature_targets[n] += row[3]*(kappa-curvature_targets[n])
        changed[n] = True
    return changed
