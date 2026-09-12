"""Submitted TF, rear-axle MPC and collision boxes share the vehicle dimensions."""
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from multi_purpose_mpc_ros import collision_geometry as cg
from multi_purpose_mpc_ros.core.wall_constraints import (
    longitudinal_extent, center_offset, wall_half_width, wall_center_bounds)

SUBMIT = Path(__file__).resolve().parents[2]


def config():
    data = yaml.safe_load((SUBMIT/'multi_purpose_mpc_ros/config/config.yaml').read_text())
    return NS(collision_geometry=NS(**data['collision_geometry']),
              bicycle_model=NS(**data['bicycle_model']))


def test_dimensions_match_vehicle_description():
    v = yaml.safe_load((SUBMIT/'racing_kart_description/config/vehicle_info.param.yaml').read_text())['/**']['ros__parameters']
    c = config()
    length = v['wheel_base'] + v['front_overhang'] + v['rear_overhang']
    assert 2*longitudinal_extent(c) == pytest.approx(length)
    assert 2*wall_half_width(c) == pytest.approx(v['wheel_tread']+v['left_overhang']+v['right_overhang'])
    assert center_offset(c) == pytest.approx(length/2-v['rear_overhang'])
    assert center_offset(c)-longitudinal_extent(c) == pytest.approx(-.510)
    assert center_offset(c)+longitudinal_extent(c) == pytest.approx(1.554)


def test_simulator_sensor_tf_matches_base_link():
    directory = SUBMIT/'racing_kart_sensor_kit_description/config'
    parent = yaml.safe_load((directory/'sensors_calibration.yaml').read_text())['base_link']['sensor_kit_base_link']
    sensors = yaml.safe_load((directory/'awsim_sensor_kit_calibration.yaml').read_text())['sensor_kit_base_link']
    assert all(v == 0 for v in parent.values())
    for name in ('gnss_link', 'imu_link'):
        assert all(v == 0 for v in sensors[name].values())


@pytest.mark.parametrize('yaw', [0., math.pi/2, math.pi, -math.pi/2])
@pytest.mark.parametrize('motion', [-.1, 0., .1])
def test_ego_center_is_fixed_in_body_frame_without_ambiguous_origin_padding(yaw, motion):
    cfg = config()
    c = NS(_collision_ego_origin=cfg.collision_geometry.ego_position_origin,
           _collision_center_offset=center_offset(cfg),
           _collision_ego_alignment=(10., 10., 1., 1.))
    pose = NS(x=10.+motion*math.cos(yaw), y=20.+motion*math.sin(yaw), theta=yaw)
    body = cg.ego_body(c, pose)
    assert (body.x, body.y) == pytest.approx(
        (pose.x+.522*math.cos(yaw), pose.y+.522*math.sin(yaw)))
    assert body.uncertainty == 0.
    assert body.lateral_padding == 0.


@pytest.mark.parametrize('yaw', [-1.5, -.6, -.2, 0., .2, .6, 1.5])
def test_rear_axle_bounds_contain_all_physical_corners(yaw):
    cfg = config()
    lower, upper = wall_center_bounds(-4.2, 4.2, course_margin=.8,
        half_width=wall_half_width(cfg), guard=.1, heading_error=yaw,
        half_length=longitudinal_extent(cfg), body_center_offset=center_offset(cfg))
    for axle_y in (lower, upper):
        body = cg.ego_body(NS(_collision_ego_origin='rear_axle'),
                           NS(x=0., y=axle_y, theta=yaw))
        for longitudinal in (-1.032, 1.032):
            for lateral in (-.725, .725):
                corner_y = body.y + longitudinal*math.sin(yaw)+lateral*math.cos(yaw)
                assert -4.9-1e-9 <= corner_y <= 4.9+1e-9
