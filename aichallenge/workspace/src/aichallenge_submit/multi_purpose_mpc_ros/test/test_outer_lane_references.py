"""Certified outer references and MPC objective integration."""
import contextlib
import ast
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from multi_purpose_mpc_ros.core.precomputed_lane_reference import (
    PrecomputedLaneReference, PrecomputedLaneReferences, design_parameters,
    apply_precomputed_heading, apply_precomputed_curvature,
)
from .probe_support import configured_mpc

ROOT = Path(__file__).parents[1]
CFG = yaml.safe_load((ROOT/'config/config.yaml').read_text())


@pytest.fixture(scope='module')
def profiles():
    m = configured_mpc()
    p = m.model.reference_path
    items = [PrecomputedLaneReference.load(
        ROOT/f'env/centerline/l{lane}_reference_wp{start}_340.csv', p, ROOT,
        design_parameters(CFG), heading_gain=1.) for lane, start in ((0,130),(2,130))]
    return PrecomputedLaneReferences(items)


@pytest.mark.parametrize('lane,start', [(0,130),(2,130)])
def test_dense_geometry_and_body_envelope(lane,start):
    generator = runpy.run_path(str(ROOT/'scripts/generate_l0_reference.py'))
    design = generator['LaneDesign'](lane_idx=lane,start_wp=start,end_wp=340)
    rows = np.genfromtxt(ROOT/f'env/centerline/l{lane}_reference_wp{start}_340.csv',delimiter=',',names=True)
    heading,kappa,metrics = design.validate(rows['e_y'])
    np.testing.assert_allclose(rows['e_psi'],heading,atol=1e-9)
    np.testing.assert_allclose(rows['kappa'],kappa,atol=1e-9)
    assert metrics['max_steering_deg'] < CFG['mpc']['delta_max_deg']
    assert metrics['max_steering_rate'] < design.max_rate
    assert metrics['min_body_wall_clearance_m'] >= .45
    assert metrics['min_body_grid_distance_m'] >= .45


@pytest.mark.parametrize('lane', [0,2])
def test_lane_lookup_heading_and_curvature_share_profile(profiles,lane):
    m=configured_mpc();p=m.model.reference_path;p.precomputed_lane_reference=profiles
    n=15;wp=305
    lateral=np.array([m._compute_lane_center(wp+i,lane) for i in range(n+1)])
    headings=np.zeros(n+1);curvature=np.zeros(n)
    apply_precomputed_heading(p,wp,lateral,headings,lane,None,False,None)
    mask=apply_precomputed_curvature(p,wp,lateral,curvature,lane,None,False,None)
    assert mask.all()
    for i in range(n):
        row=profiles.sample(wp+i,lane)
        assert lateral[i] == row[0]
        assert headings[i] == pytest.approx(row[1])
        assert curvature[i] == pytest.approx(row[2])
    assert profiles.sample(312,1) is None
    assert profiles.sample(341,lane) is None


@pytest.mark.parametrize('lane,soft,has_targets,weights', [
    (None,None,False,None),(1,None,False,None),(None,0,True,None),
    (None,2,True,None),(None,1,True,np.ones(16))])
def test_unrelated_objectives_untouched(profiles,lane,soft,has_targets,weights):
    p=configured_mpc().model.reference_path;p.precomputed_lane_reference=profiles
    h=np.full(16,.123);k=np.full(15,.234)
    apply_precomputed_heading(p,305,np.zeros(16),h,lane,soft,has_targets,weights)
    mask=apply_precomputed_curvature(p,305,np.zeros(16),k,lane,soft,has_targets,weights)
    assert not mask.any()
    np.testing.assert_array_equal(h,np.full(16,.123))
    np.testing.assert_array_equal(k,np.full(15,.234))


@pytest.mark.parametrize('lane', [0,2])
def test_objectives_do_not_modify_dynamics_or_walls(profiles,lane):
    m=configured_mpc();p=m.model.reference_path;p.target_lane_idx=lane
    wp=p.get_waypoint(312);row=profiles.sample(312,lane)
    m.model.update_states(wp.x-row[0]*np.sin(wp.psi),wp.y+row[0]*np.cos(wp.psi),wp.psi+row[1])
    m.model.wp_id=312;m.model.spatial_state=m.model.t2s(wp,m.model.temporal_state)
    with contextlib.redirect_stdout(io.StringIO()):m._init_problem(m.N,0.)
    A=m.A0.toarray().copy();lo=m.optimizer._derivative_cache['l'].copy();hi=m.optimizer._derivative_cache['u'].copy();q=m.optimizer._derivative_cache['q'].copy()
    p.precomputed_lane_reference=profiles;p.precomputed_curvature_enabled=True
    m.osqp_initialized=False
    with contextlib.redirect_stdout(io.StringIO()):m._init_problem(m.N,0.)
    np.testing.assert_allclose(m.A0.toarray(),A)
    np.testing.assert_allclose(m.optimizer._derivative_cache['l'],lo)
    np.testing.assert_allclose(m.optimizer._derivative_cache['u'],hi)
    assert not np.allclose(m.optimizer._derivative_cache['q'],q)


def test_hybrid_uses_blended_geometry_not_final_lane(profiles):
    m=configured_mpc();p=m.model.reference_path;p.precomputed_lane_reference=profiles
    # Both transitions currently share the Center reference (zero spatial weight).
    # Their heading and curvature must agree, despite different destination lanes.
    results=[]
    for lane in (0,2):
        h=np.zeros(16);k=np.zeros(15)
        apply_precomputed_heading(p,305,np.zeros(16),h,lane,lane,True,np.zeros(16))
        apply_precomputed_curvature(p,305,np.zeros(16),k,lane,lane,True,np.zeros(16))
        results.append((h,k))
    np.testing.assert_allclose(results[0][0],results[1][0])
    np.testing.assert_allclose(results[0][1],results[1][1])
    assert not np.allclose(results[0][1],[profiles.sample(305+i,2)[2] for i in range(15)])


@pytest.mark.parametrize('lane,start', [(0,130),(2,130)])
def test_installed_loader_checks_profile_and_current_source(profiles, tmp_path, lane, start):
    (tmp_path/'env').symlink_to((ROOT/'env').resolve(), target_is_directory=True)
    relative = f'env/centerline/l{lane}_reference_wp{start}_340.csv'
    path = configured_mpc().model.reference_path
    profile = PrecomputedLaneReference.load(
        tmp_path/relative, path, tmp_path, design_parameters(CFG))
    assert profile.sample(305,lane) is not None
    # A valid checksum on the CSV does not excuse stale source metadata.
    copied = tmp_path/'profile.csv'
    copied.write_bytes((ROOT/relative).read_bytes())
    metadata = json.loads((ROOT/relative).with_suffix('.json').read_text())
    metadata['sources']['multi_purpose_mpc_ros/core/reference_path.py'] = 'stale'
    copied.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='source changed'):
        PrecomputedLaneReference.load(copied,path,tmp_path,design_parameters(CFG))
    metadata = json.loads((ROOT/relative).with_suffix('.json').read_text())
    copied.with_suffix('.json').write_text(json.dumps(metadata))
    copied.write_text(copied.read_text()+'\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        PrecomputedLaneReference.load(copied,path,tmp_path,design_parameters(CFG))


def test_controller_startup_loads_both_lanes_from_installed_share(tmp_path):
    (tmp_path/'env').symlink_to((ROOT/'env').resolve(), target_is_directory=True)
    path = configured_mpc().model.reference_path
    controller = SimpleNamespace(
        _cfg=CFG, _reference_pathN_center=path,
        in_pkg_share=lambda relative: str(tmp_path/relative),
        get_logger=Mock(return_value=Mock()))
    tree = ast.parse((ROOT/'multi_purpose_mpc_ros/mpc_controller.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MPCController')
    init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_initialize')
    def assigns(node, name):
        return isinstance(node,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id==name for t in node.targets)
    first = next(i for i,n in enumerate(init.body) if assigns(n,'profile_csv'))
    last = next(i for i,n in enumerate(init.body) if assigns(n,'unsafe_static_fallback'))
    exec(compile(ast.Module(body=init.body[first:last],type_ignores=[]),'<startup>','exec'),
         dict(self=controller,cfg_ref_path=SimpleNamespace(**CFG['reference_path'])))
    assert isinstance(path.precomputed_lane_reference,PrecomputedLaneReferences)
    assert set(path.precomputed_lane_reference.profiles)=={0,2}
    assert path.precomputed_curvature_enabled
    controller.get_logger().error.assert_not_called()


@pytest.mark.parametrize('lane,index', [(0,144),(0,150),(0,162),(0,194),(0,205),
                                       (2,149),(2,174),(2,180),(2,203),(2,225),(2,249)])
def test_extended_reference_resolves_corner_failures(profiles, lane, index):
    m = configured_mpc()
    p = m.model.reference_path
    p.precomputed_lane_reference = profiles
    p.precomputed_curvature_enabled = True
    p.target_lane_idx = lane
    p.is_overtaking = True
    def point(i):
        wp = p.get_waypoint(i)
        ey = m._compute_lane_center(i, lane)
        return np.array([wp.x-ey*np.sin(wp.psi), wp.y+ey*np.cos(wp.psi)])
    xy = point(index)
    direction = point(index+1)-point(index-1)
    yaw = np.arctan2(direction[1], direction[0])
    wp = p.get_waypoint(index)
    steer = np.clip(np.arctan(m.model.length*wp.kappa*(1+m.understeer_coeff*(35/3.6)**2)),
                    -np.deg2rad(18), np.deg2rad(18))
    m.current_control = np.tile([35/3.6, steer], m.N)
    for _ in range(3):
        m.model.update_states(*xy, yaw)
        m.previous_steering = steer
        m.get_control()
    assert m.last_solution_accurate, m.failure_reason
