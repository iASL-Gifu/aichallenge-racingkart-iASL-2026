"""Geometry reuse retains conservative checks and display cannot gate safety."""
import math
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from multi_purpose_mpc_ros import collision_geometry as cg
from multi_purpose_mpc_ros.core import reference_path as rp
from .test_overtake_session import controller_method


def test_display_defaults_off_and_enabled_rate_is_bounded():
    clock = NS(nanoseconds=0)
    class Marker:
        DELETEALL = 3
    call = controller_method('_publish_collision_bodies')
    call.__globals__['MarkerArray'] = lambda: NS(markers=[])
    call.__globals__['Marker'] = Marker
    c = NS(_cfg=NS(mpc=NS()),
           get_clock=lambda: NS(now=lambda: clock),
           _v2x_tracker=NS(active_vehicle_ids=lambda: []),
           _collision_body_publisher=Mock())
    pose = NS(x=0., y=0., theta=0.)
    call(c, pose)
    c._collision_body_publisher.publish.assert_not_called()
    c._cfg.mpc.collision_body_visualization_enabled = True
    for time in [0., .025, .1, .19, .201]:
        clock.nanoseconds = int(time*1e9)
        call(c, pose)
    assert c._collision_body_publisher.publish.call_count == 2
    clock.nanoseconds = 0  # simulation reset must not suppress new display
    call(c, pose)
    assert c._collision_body_publisher.publish.call_count == 3


def test_swept_samples_reused_across_targets_without_reusing_verdict(monkeypatch):
    # Exercise narrow-phase sample sharing independently of the far-target filter.
    monkeypatch.setattr(cg, '_segment_definitely_separated', lambda *args: False)
    cg._sweep_ego_samples.cache_clear()
    a = cg.BodyPose(0., 0., 0., 0.)
    b = replace(a, x=1., yaw=.1)
    geometry = cg.BodyGeometry()
    far = replace(a, y=10.)
    assert cg.swept_path_clear([a,b], [0.,1.], far, (0.,0.), geometry)
    assert cg.swept_path_clear([a,b], [0.,1.], replace(far, y=12.), (1.,0.), geometry)
    assert cg._sweep_ego_samples.cache_info().hits == 1
    assert not cg.swept_path_clear([a,b], [0.,1.], a, (0.,0.), geometry)
    assert cg._sweep_ego_samples.cache_info().hits == 2
    assert not cg.swept_path_clear([a,b], [0.,1.], replace(a, position_valid=False), (0.,0.), geometry)
    cg.swept_path_clear([a,replace(b, uncertainty=.2)], [0.,1.], far, (0.,0.), geometry)
    assert cg._sweep_ego_samples.cache_info().misses == 2


def test_boundary_cached_scalar_math_matches_original_and_invalidates():
    rng = np.random.default_rng(230)
    rp._signed_boundary_distance.cache_clear()
    for x, y, psi, bx, by in rng.uniform(-100, 100, (100, 5)):
        angle = np.mod(np.arctan2(by-y,bx-x)-psi+math.pi,2*math.pi)-math.pi
        expected = np.sign(angle)*np.sqrt((bx-x)**2+(by-y)**2)
        assert rp._signed_boundary_distance(x,y,psi,bx,by) == expected
        assert rp._signed_boundary_distance(x,y,psi,bx,by) == expected
    assert rp._signed_boundary_distance.cache_info().hits == 100
    assert rp._signed_boundary_distance(0.,0.,0.,0.,1.) == 1.
    assert rp._signed_boundary_distance(0.,0.,0.,0.,-1.) == -1.
