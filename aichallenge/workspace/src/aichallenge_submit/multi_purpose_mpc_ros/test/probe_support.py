"""ROS-free construction using the submitted map, trajectories and MPC settings."""
from contextlib import redirect_stdout
import io
import math
from pathlib import Path

import numpy as np
from scipy import sparse
import yaml

from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC


def configured_mpc(kind='center'):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'config/config.yaml').read_text())
    ref, c, b = cfg['reference_path'], cfg['mpc'], cfg['bicycle_model']
    trajectory = np.genfromtxt(root / ref[f'{kind}_csv_path'], delimiter=',', names=True)
    with redirect_stdout(io.StringIO()):
        road = Map(str(root / cfg['map']['yaml_path']))
        path = ReferencePath(
            road, trajectory['x_m'], trajectory['y_m'],
            ref['resolution'], ref['smoothing_distance'], ref['max_width'],
            ref['circular'], wp_psi=trajectory['psi_rad'],
            bounds_csv_path=str(root / ref[f'{kind}_bounds_csv_path']))
        # Physical margins are ready before any vector-map callback.
        path.compute_speed_profile(dict(a_min=c['a_min'], a_max=c['a_max'],
                                        v_min=0., v_max=c['v_max']/3.6, ay_max=c['ay_max']))
    path.unsafe_static_fallback_on_narrow = ref['unsafe_static_fallback_on_narrow']
    car = BicycleModel(path, b['length'], b['width'], 1.0/c['control_rate'])
    limit = math.tan(math.radians(c['delta_max_deg'])) / car.length
    mpc = MPC(car, c['N'], sparse.diags(c['Q']), sparse.diags(c['R']), sparse.diags(c['QN']),
              dict(xmin=np.full(3, -np.inf), xmax=np.full(3, np.inf)),
              dict(umin=np.array([0., -limit]), umax=np.array([c['v_max']/3.6, limit])),
              c['ay_max'], c['steer_rate_max']/c['steering_tire_angle_gain_var'], 0,
              True, False, c['use_max_kappa_pred'], c['understeer_coeff'],
              steering_command_delay=c['steering_command_delay'],
              steering_reservation_enabled=c.get('steering_reservation_enabled', False))
    for name in ('solve_time_budget_ms', 'max_prediction_fallback_cycles',
                 'prediction_outer_boundary_guard', 'prediction_lateral_tolerance',
                 'lane_constraint_retry_relaxation_m', 'lane_constraint_retry_relax_toward_center_only',
                 'lane_constraint_retry_taper_over_horizon', 'lane_constraint_retry_terminal_ratio',
                 'lane_constraint_connection_points'):
        if name in c:
            setattr(mpc, name, c[name])
    mpc.debug_counter = 1
    return mpc
