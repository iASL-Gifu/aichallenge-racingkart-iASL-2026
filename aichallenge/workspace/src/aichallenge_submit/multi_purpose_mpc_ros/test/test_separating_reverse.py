"""Reverse may leave clearance overlap, but never approach or touch a body."""
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch

import pytest
from multi_purpose_mpc_ros import collision_geometry as cg
from .test_overtake_session import controller_method

G = cg.BodyGeometry()


def poses():
    return [cg.BodyPose(-i*.02, 0., 0., 0.) for i in range(21)]


def check(path, target, velocity=(0., 0.)):
    return cg.separating_path_clear(
        path, [i*.1 for i in range(len(path))], target, velocity, G, reverse=True)


def test_reverse_leaves_front_vehicle_margin_overlap():
    target = cg.BodyPose(G.length+.06, 0., 0., 0.)
    path = poses()
    assert not cg.swept_path_clear(path, [i*.1 for i in range(len(path))], target, (0., 0.), G)
    assert check(path, target)
    assert not cg.separating_forward_path_clear(
        path, [i*.1 for i in range(len(path))], target, (0., 0.), G)


@pytest.mark.parametrize('target', [
    cg.BodyPose(G.length-.01, 0., 0., 0.),  # real contact
    cg.BodyPose(G.length+.06, 0., 0., 0., uncertainty=.1),
    cg.BodyPose(G.length+.06, 0., None, 0.),
    cg.BodyPose(-G.length-.06, 0., 0., 0.),  # backing into rear car
])
def test_reverse_rejects_contact_uncertainty_unknown_yaw_and_rear_car(target):
    assert not check(poses(), target)


def test_reverse_rejects_closing_moving_front_vehicle():
    assert not check(poses(), cg.BodyPose(G.length+.06, 0., 0., 0.), (-.3, 0.))


def test_reverse_rejects_turning_front_corner_contact():
    path = [cg.BodyPose(-i*.01, 0., i*.15, 0.) for i in range(4)]
    assert not check(path, cg.BodyPose(G.length+.06, 0., 0., 0.))


def test_reverse_rejects_insufficient_separation_and_direction_change():
    target = cg.BodyPose(G.length+.06, 0., 0., 0.)
    assert not check(poses()[:2], target)
    path = poses() + [cg.BodyPose(-.39, 0., 0., 0.)]
    assert not check(path, target)


def test_controller_reverse_still_checks_wall_and_every_vehicle():
    c = NS(_recovery_localization_available=True, _straight_reentry_speed=1.,
           _reverse_overlap_allowance=.03, _collision_now=0., _collision_ego_origin='center',
           _map=NS(static_recovery_path_is_clear=lambda *a, **kw: (True, 'clear')),
           _v2x_tracker=NS(active_vehicle_ids=lambda: ['front'],
                           has_velocity_estimate=lambda vid: True,
                           velocity=lambda vid: (0., 0.)))
    c.check = MethodType(controller_method('_reentry_path_is_clear'), c)
    path = [(p.x, p.y, p.yaw) for p in poses()]
    times = [i*.1 for i in range(len(path))]
    targets = {'front': cg.BodyPose(G.length+.06, 0., 0., 0.),
               'rear': cg.BodyPose(-G.length-.2, 0., 0., 0.)}
    with patch.object(cg, 'target_body', side_effect=lambda c, vid: targets[vid]):
        assert c.check(path, times=times, reverse=True)[0]
        c._v2x_tracker.active_vehicle_ids = lambda: ['front', 'rear']
        assert c.check(path, times=times, reverse=True) == (False, 'vehicle_collision=rear')
        c._map.static_recovery_path_is_clear = lambda *a, **kw: (False, 'wall')
        assert c.check(path, times=times, reverse=True) == (False, 'wall')
