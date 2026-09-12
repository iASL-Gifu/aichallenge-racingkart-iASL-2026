#!/usr/bin/env python3
"""Offline outer-lane objective generation. Never called by the ROS control loop.

Run from the source package; --check validates the committed CSV without solving.
The Center coordinate basis and all road/lane/obstacle bounds are preserved.
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multi_purpose_mpc_ros.core.precomputed_lane_reference import file_digest, design_parameters


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class LaneDesign:
    def __init__(self, root=ROOT, lane_idx=0, start_wp=220, end_wp=290):
        self.lane_idx = lane_idx
        self.start_wp, self.end_wp = start_wp, end_wp
        self.root = Path(root)
        self.cfg = yaml.safe_load((self.root / 'config/config.yaml').read_text())
        ref_cfg = self.cfg['reference_path']
        self.sources = [ref_cfg['center_csv_path'], ref_cfg['center_bounds_csv_path'],
                        'multi_purpose_mpc_ros/core/reference_path.py', self.cfg['map']['yaml_path']]
        trajectory = np.genfromtxt(self.root / self.sources[0], delimiter=',', names=True)
        bounds = np.genfromtxt(self.root / self.sources[1], delimiter=',', names=True)
        self.world = np.c_[trajectory['x_m'], trajectory['y_m']]
        self.origin = self.world[260].copy()
        self.p = self.world - self.origin
        d = np.roll(self.p, -1, axis=0) - self.p
        self.yaw = np.arctan2(d[:, 1], d[:, 0])
        self.normal = np.c_[-np.sin(self.yaw), np.cos(self.yaw)]
        # Execute the production lane partition without importing ROS/ament.
        tree = ast.parse((self.root / self.sources[2]).read_text())
        self.margin = next(ast.literal_eval(n.value) for n in tree.body
                           if isinstance(n, ast.Assign) and any(
                               isinstance(k, ast.Name) and k.id == 'OUTER_COURSE_MARGIN' for k in n.targets))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ReferencePath')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_lane_bounds')
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<production-lane-bounds>', 'exec'), ns)
        self.lb, self.ub = bounds['lb'] + self.margin, bounds['ub'] - self.margin
        ref = SimpleNamespace(n_lanes=3, inner_lane_width=.5,
                              get_waypoint=lambda i: SimpleNamespace(lb=self.lb[i], ub=self.ub[i]))
        self.lanes = np.array([ns['get_lane_bounds'](ref, i) for i in range(len(self.p))])
        self.base = self.lanes[:, self.lane_idx].mean(axis=1)
        if lane_idx not in (0, 2) or not 0 <= start_wp < end_wp < len(self.p) or end_wp-start_wp < 20:
            raise ValueError('outer lane and at least 21 non-wrapping waypoints required')
        self.ids = np.arange(start_wp, end_wp + 1)
        self.parameters = design_parameters(self.cfg)
        self.wheelbase = self.cfg['bicycle_model']['length']
        self.speed = self.cfg['mpc']['v_max'] / 3.6
        self.understeer = self.cfg['mpc']['understeer_coeff']
        self.max_delta = np.deg2rad(self.cfg['mpc']['delta_max_deg'])
        self.max_rate = self.cfg['mpc']['steer_rate_max'] / self.cfg['mpc']['steering_tire_angle_gain_var']
        body = self.cfg['collision_geometry']
        # Union of center- and rear-axle-origin footprints. No live origin setting is changed.
        self.rear = -body['length'] / 2
        self.front = body['length'] / 2 + body['rear_axle_to_center']
        self.half_width = max(body['width'], self.cfg['bicycle_model']['width']) / 2
        self.walls = []
        for name, sign, bound in [('right', 1, bounds['lb']), ('left', -1, bounds['ub'])]:
            # Check both surveyed CSV wall points and the production reconstructed wall.
            for wall in (np.c_[bounds[name + '_x'], bounds[name + '_y']] - self.origin,
                         self.p + self.normal * bound[:, None]):
                wall = wall[max(0, start_wp-10):end_wp+11]
                self.walls.append((wall[:-1], np.diff(wall, axis=0), sign))
        map_file = self.root / self.cfg['map']['yaml_path']
        self.map_cfg = yaml.safe_load(map_file.read_text())
        image_file = map_file.parent / self.map_cfg['image']
        self.sources.append(str(image_file.relative_to(self.root)))
        pixels = np.asarray(Image.open(image_file))
        occupancy = pixels / 255.0 if self.map_cfg['negate'] else 1.0 - pixels / 255.0
        self.map_distance = distance_transform_edt(occupancy < self.map_cfg['free_thresh']) * self.map_cfg['resolution']

    def points(self, x):
        return self.p[self.ids] + self.normal[self.ids] * x[:, None]

    def geometry(self, x):
        q = self.points(x)
        d = np.diff(q, axis=0)
        ds = np.linalg.norm(d, axis=1)
        k = 2 * cross(d[:-1], d[1:]) / (ds[:-1] * ds[1:] * np.linalg.norm(d[:-1] + d[1:], axis=1))
        heading = np.arctan2(d[:, 1], d[:, 0])
        steering = np.arctan(self.wheelbase * k * (1 + self.understeer * self.speed**2))
        return q, heading, k, steering, ds

    def body_points(self, q, heading, dense=False):
        f = np.c_[np.cos(heading), np.sin(heading)]
        n = np.c_[-f[:, 1], f[:, 0]]
        along = np.linspace(self.rear, self.front, 28) if dense else [self.rear, self.front]
        lateral = np.linspace(-self.half_width, self.half_width, 18) if dense else [-self.half_width, self.half_width]
        return np.array([q + a * f + b * n for a in along for b in lateral]).reshape(-1, 2)

    def clearance(self, points):
        result = []
        for a, d, sign in self.walls:
            for chunk in np.array_split(points, max(1, len(points) // 1500)):
                rel = chunk[:, None, :] - a
                u = np.clip(np.sum(rel * d, axis=2) / np.sum(d * d, axis=1), 0, 1)
                residual = rel - u[:, :, None] * d
                idx = np.linalg.norm(residual, axis=2).argmin(axis=1)
                # Signed Euclidean distance also handles concave wall vertices.
                signed = sign * cross(d[idx], rel[np.arange(len(chunk)), idx])
                result.extend((np.sign(signed) * np.linalg.norm(residual[np.arange(len(chunk)), idx], axis=1)).tolist())
        return np.array(result)

    def constraints(self, x, dense=True):
        q, h, k, delta, ds = self.geometry(x)
        limits = np.r_[.25 - np.abs(k),
                       .9 * self.max_rate - np.abs(np.diff(delta)) * self.speed / ds[1:-1],
                       self.clearance(self.body_points(q[:-1], h)) - .30]
        if not dense:
            return limits
        # Constrain the same interpolated curve that is certified at export.
        # Knot-only rate limits can miss steering peaks between waypoints.
        _, _, _, _, dense_delta, dense_rate = self.dense_geometry(q, ds)
        return np.r_[limits,
                     .98 * self.max_delta - np.abs(dense_delta),
                     .95 * self.max_rate - dense_rate]

    def dense_geometry(self, q, ds):
        s = np.r_[0, np.cumsum(ds)]
        curve = CubicSpline(s, q)
        # Fixed sample count keeps SLSQP's constraint vector size constant.
        # Each interval is at most .05 m for any point inside guarded L0.
        ids = self.ids
        lo = self.points(self.lanes[ids, self.lane_idx, 1] + .1)
        hi = self.points(self.lanes[ids, self.lane_idx, 0] - .05)
        max_ds = np.maximum.reduce([
            np.linalg.norm(b[1:] - a[:-1], axis=1)
            for a in (lo, hi) for b in (lo, hi)])
        counts = np.maximum(1, np.ceil(max_ds / .05).astype(int))
        sd = np.r_[np.concatenate([
            np.linspace(s[i], s[i+1], count, endpoint=False)
            for i, count in enumerate(counts)]), s[-1]]
        qd, d1, d2 = curve(sd), curve(sd, 1), curve(sd, 2)
        kd = cross(d1, d2) / np.linalg.norm(d1, axis=1)**3
        delta = np.arctan(self.wheelbase * kd * (1 + self.understeer * self.speed**2))
        rate = np.abs(np.diff(delta)) * self.speed / np.linalg.norm(np.diff(qd, axis=0), axis=1)
        return curve, qd, d1, kd, delta, rate

    def objective(self, x):
        q, h, k, delta, ds = self.geometry(x)
        return np.sum((x - self.base[self.ids])**2) + 10 * np.sum(np.diff(k)**2) + np.sum(k*k)

    def solve(self, seed=None):
        original = self.base[self.ids]
        bounds = list(zip(self.lanes[self.ids, self.lane_idx, 1] + .1, self.lanes[self.ids, self.lane_idx, 0] - .05))
        for i in [0, 1, 2, len(original)-3, len(original)-2, len(original)-1]:
            bounds[i] = (original[i], original[i])
        if seed is not None:
            seed = np.asarray(seed, dtype=float)
            if seed.shape != original.shape or not np.isfinite(seed).all():
                raise ValueError('invalid initial reference')
            limits = np.asarray(bounds)
            initial = SimpleNamespace(
                x=np.clip(seed, limits[:, 0], limits[:, 1]),
                message='using existing profile inside current lane bounds')
        else:
            # First remove sharp midpoint kinks before optimizing the dense envelope.
            initial = minimize(self.objective, original, method='SLSQP', bounds=bounds,
                               constraints=[{'type': 'ineq', 'fun': lambda x: self.constraints(x, dense=False)}],
                               options={'maxiter': 400, 'ftol': 1e-6})
            if min(self.constraints(initial.x, dense=False)) < -1e-5:
                # A kinked midpoint can leave SLSQP without a useful first step.
                # Start again from a bounded spatial bending-energy minimizer.
                smooth = minimize(
                    lambda z: np.sum(np.diff(self.points(z), n=2, axis=0)**2)
                        + .01*np.sum((z-original)**2),
                    original, method='SLSQP', bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-8})
                initial = minimize(self.objective, smooth.x, method='SLSQP', bounds=bounds,
                                   constraints=[{'type': 'ineq', 'fun': lambda x: self.constraints(x, dense=False)}],
                                   options={'maxiter': 400, 'ftol': 1e-6})
        print(f'L{self.lane_idx} initial: {initial.message}; min={min(self.constraints(initial.x, dense=False)):.6f}', flush=True)
        result = minimize(self.objective, initial.x, method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'ineq', 'fun': self.constraints}],
                          options={'maxiter': 400, 'ftol': 1e-6})
        if min(self.constraints(result.x)) < -1e-5:
            raise RuntimeError(f'No feasible L{self.lane_idx} reference: {result.message}; min_constraint={min(self.constraints(result.x)):.6f}')
        # Feasibility and dense validation, not optimizer status, authorize export.
        return result.x

    def validate(self, x):
        q, h, k, delta, ds = self.geometry(x)
        if not np.isfinite(x).all() or min(self.constraints(x)) < -1e-5:
            raise ValueError('discrete curvature/rate/body constraints failed')
        if (np.any(x < self.lanes[self.ids, self.lane_idx, 1] + .1 - 1e-6)
                or np.any(x > self.lanes[self.ids, self.lane_idx, 0] - .05 + 1e-6)):
            raise ValueError('reference outside guarded outer lane')
        if not np.allclose(x[[0, 1, 2, -3, -2, -1]], self.base[self.ids][[0, 1, 2, -3, -2, -1]], atol=1e-7, rtol=0):
            raise ValueError('reference seams changed')
        # Cubic reconstruction at <= 5 cm spacing; sweep the complete rectangle.
        s = np.r_[0, np.cumsum(ds)]
        curve, qd, d1, kd, delta_d, rate = self.dense_geometry(q, ds)
        bodies = self.body_points(qd, np.arctan2(d1[:, 1], d1[:, 0]), dense=True)
        # Perimeter-to-wall checks and complete interior-to-occupancy checks.
        f = np.c_[np.cos(np.arctan2(d1[:, 1], d1[:, 0])), np.sin(np.arctan2(d1[:, 1], d1[:, 0]))]
        n = np.c_[-f[:, 1], f[:, 0]]
        perimeter = np.array([qd + a*f + b*n for a in np.linspace(self.rear, self.front, 28)
                              for b in [-self.half_width, self.half_width]]).reshape(-1, 2)
        clearance = float(self.clearance(perimeter).min())
        res = self.map_cfg['resolution']
        pix = np.floor((bodies + self.origin - np.array(self.map_cfg['origin'][:2])) / res).astype(int)
        height, width = self.map_distance.shape
        if np.any(pix < 0) or np.any(pix[:, 0] >= width) or np.any(pix[:, 1] >= height):
            raise ValueError('body sweep outside occupancy map')
        grid_clearance = float(self.map_distance[height - 1 - pix[:, 1], pix[:, 0]].min())
        if np.max(np.abs(delta_d)) > self.max_delta or np.max(rate) > self.max_rate:
            raise ValueError('dense interpolation exceeds steering envelope')
        if clearance < .25 or grid_clearance < .25:
            raise ValueError(f'dense body sweep hits wall clearance: {clearance}, grid={grid_clearance}')
        knots_d1, knots_d2 = curve(s, 1), curve(s, 2)
        yaw = np.arctan2(knots_d1[:, 1], knots_d1[:, 0])
        e_psi = (yaw - self.yaw[self.ids] + np.pi) % (2 * np.pi) - np.pi
        curvature = cross(knots_d1, knots_d2) / np.linalg.norm(knots_d1, axis=1)**3
        metrics = dict(max_curvature=float(np.max(np.abs(kd))), max_steering_deg=float(np.rad2deg(np.max(np.abs(delta_d)))),
                       max_steering_rate=float(np.max(rate)), design_speed_mps=self.speed,
                       min_body_wall_clearance_m=clearance, min_body_grid_distance_m=grid_clearance,
                       sweep_step_m=.05, body_width_m=self.half_width*2,
                       body_longitudinal_interval_m=[self.rear, self.front])
        return e_psi, curvature, metrics

    def export(self, path, x):
        e_psi, curvature, metrics = self.validate(x)
        t = np.minimum(np.clip((self.ids-self.start_wp)/10, 0, 1), np.clip((self.end_wp-self.ids)/10, 0, 1))
        weight = t**3 * (10 - 15*t + 6*t*t)
        table = np.c_[self.ids, self.world[self.ids], self.yaw[self.ids], self.lb[self.ids], self.ub[self.ids], x, e_psi, curvature, weight]
        np.savetxt(path, table, delimiter=',', comments='', fmt='%.12f',
                   header='wp_id,base_x,base_y,base_psi,base_lb,base_ub,e_y,e_psi,kappa,weight')
        meta = dict(version=1, lane_idx=self.lane_idx, parameters=self.parameters, profile_sha256=file_digest(path),
                    sources={s: file_digest(self.root/s) for s in self.sources}, metrics=metrics)
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--lane', type=int, choices=(0, 2), default=0)
    parser.add_argument('--start-wp', type=int, default=220)
    parser.add_argument('--end-wp', type=int, default=290)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--revalidate-from', type=Path,
                        help='Revalidate an existing lateral profile and export with current design metadata')
    parser.add_argument('--initial-reference', type=Path,
                        help='Use an existing profile as a seed for constrained re-optimization')
    args = parser.parse_args()
    if sum(bool(value) for value in (args.check, args.revalidate_from, args.initial_reference)) > 1:
        parser.error('--check, --revalidate-from and --initial-reference are mutually exclusive')
    design = LaneDesign(lane_idx=args.lane, start_wp=args.start_wp, end_wp=args.end_wp)
    path = args.output or ROOT / f'env/centerline/l{args.lane}_reference_wp{args.start_wp}_{args.end_wp}.csv'
    start = time.perf_counter()
    if args.check or args.revalidate_from or args.initial_reference:
        rows = np.atleast_1d(np.genfromtxt(
            args.revalidate_from or args.initial_reference or path, delimiter=',', names=True))
        if not np.array_equal(rows['wp_id'], design.ids):
            raise ValueError('profile waypoint range does not match requested design')
        x = rows['e_y']
    if args.check:
        _, _, metrics = design.validate(x)
    elif args.revalidate_from:
        metrics = design.export(path, x)
    else:
        metrics = design.export(path, design.solve(x if args.initial_reference else None))
    print(json.dumps(metrics, indent=2))
    print(f'elapsed_s={time.perf_counter()-start:.3f}')


if __name__ == '__main__':
    main()
