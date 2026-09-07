"""Offline geometry certification and objective-only MPC integration."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy

import numpy as np
import pytest
import yaml

from multi_purpose_mpc_ros.core.MPC import spatial_lane_transition_reference
from multi_purpose_mpc_ros.core.precomputed_lane_reference import (
    PrecomputedLaneReference, apply_precomputed_heading, design_parameters,
)
from .probe_support import configured_mpc
from .test_overtake_session import controller_method

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'config/config.yaml').read_text())
CSV = ROOT / CFG['reference_path']['l0_precomputed_reference_csv']


@pytest.fixture(scope='module')
def configured():
    mpc = configured_mpc()
    path = mpc.model.reference_path
    profile = PrecomputedLaneReference.load(CSV, path, ROOT, design_parameters(CFG))
    return mpc, path, profile


def test_offline_sweep_and_steering_envelope():
    ns = runpy.run_path(str(ROOT / 'scripts/generate_l0_reference.py'))
    design = ns['LaneDesign']()
    rows = np.genfromtxt(CSV, delimiter=',', names=True)
    _, _, metrics = design.validate(rows['e_y'])
    assert metrics['max_steering_deg'] < CFG['mpc']['delta_max_deg']
    assert metrics['min_body_wall_clearance_m'] >= .25
    assert metrics['min_body_grid_distance_m'] >= .25


def test_loader_rejects_other_path_and_design(configured):
    _, path, profile = configured
    changed = design_parameters(CFG)
    changed['bicycle_model.width'] += .1
    with pytest.raises(ValueError, match='parameters changed'):
        PrecomputedLaneReference.load(CSV, path, ROOT, changed)
    wp = path.get_waypoint(260)
    before = wp.x
    try:
        wp.x += .01
        with pytest.raises(ValueError, match='geometry mismatch'):
            PrecomputedLaneReference.load(CSV, path, ROOT, design_parameters(CFG))
    finally:
        wp.x = before


def test_profile_only_l0_and_wraps_without_changing_seams(configured):
    mpc, path, profile = configured
    path.precomputed_lane_reference = profile
    for wp in (219, 220, 290, 291):
        assert profile.sample(wp, 0) is None
    for lane in (None, 1, 2):
        assert profile.sample(260, lane) is None
    assert mpc._compute_lane_center(260, 0) == profile.sample(260, 0)[0]
    assert profile.sample(path.n_waypoints + 260, 0)[0] == profile.sample(260, 0)[0]
    with pytest.raises(ValueError):
        profile.table[260, 0] = 0.0


@pytest.mark.parametrize('lane,soft,has_targets,weights', [
    (None, None, False, None), (1, None, False, None), (2, None, False, None),
    (None, None, True, None), (None, 1, True, np.ones(4)),
    (0, 1, True, np.ones(4)),
])
def test_other_objectives_keep_their_heading(configured, lane, soft, has_targets, weights):
    _, path, profile = configured
    path.precomputed_lane_reference = profile
    headings = np.full(5, .123)
    apply_precomputed_heading(path, 250, np.zeros(5), headings, lane, soft, has_targets, weights)
    np.testing.assert_array_equal(headings, .123)


def test_hybrid_uses_blended_lateral_reference_not_completed_lane(configured):
    mpc, path, profile = configured
    path.precomputed_lane_reference = profile
    lateral, weights = spatial_lane_transition_reference(np.arange(5.), 0.,
        [mpc._compute_lane_center(250+i, 0) for i in range(5)], 15.)
    headings = np.zeros(5)
    apply_precomputed_heading(path, 250, lateral, headings, 0, 0, True, weights[1:])
    assert not np.allclose(headings, [profile.sample(250+i, 0)[1] for i in range(5)])
    a, b = path.get_waypoint(250), path.get_waypoint(251)
    dx = b.x - lateral[1]*np.sin(b.psi) - a.x + lateral[0]*np.sin(a.psi)
    dy = b.y + lateral[1]*np.cos(b.psi) - a.y - lateral[0]*np.cos(a.psi)
    expected = (np.arctan2(dy, dx) - a.psi + np.pi) % (2*np.pi) - np.pi
    assert headings[0] == pytest.approx(profile.heading_gain * expected)


def test_real_qp_changes_only_objective_and_shadow_preserves_live_path(configured):
    mpc, path, profile = configured
    wp_id = 260
    wp = path.get_waypoint(wp_id)
    sample = profile.sample(wp_id, 0)
    mpc.model.update_states(wp.x-sample[0]*np.sin(wp.psi), wp.y+sample[0]*np.cos(wp.psi), wp.psi+sample[1])
    mpc.model.wp_id = wp_id
    mpc.model.spatial_state = mpc.model.t2s(wp, mpc.model.temporal_state)
    path.target_lane_idx = 0
    mpc.set_soft_lateral_reference()
    mpc.set_lane_transition_weights()
    snapshots = []
    for active in (None, profile):
        path.precomputed_lane_reference = active
        with redirect_stdout(io.StringIO()):
            mpc._init_problem(mpc.N, 0.)
        cache = mpc.optimizer._derivative_cache
        snapshots.append({key: cache[key].copy() for key in ('P', 'A', 'q', 'l', 'u')})
    for key in ('P', 'A'):
        np.testing.assert_array_equal(snapshots[0][key].toarray(), snapshots[1][key].toarray())
    for key in ('l', 'u'):
        np.testing.assert_array_equal(snapshots[0][key], snapshots[1][key])
    assert not np.array_equal(snapshots[0]['q'], snapshots[1]['q'])
    before = profile.table.copy()
    with pytest.raises(RuntimeError):
        with controller_method('_probe_corridor')(None, mpc.model, 0) as probe:
            assert probe.precomputed_lane_reference.sample(260, 0)[0] == sample[0]
            raise RuntimeError('probe failed')
    assert mpc.model.reference_path is path
    np.testing.assert_array_equal(profile.table, before)


@pytest.mark.parametrize('wp_id', range(256, 281))
def test_permitted_s_curve_l0_qp_solves(configured, wp_id):
    mpc, path, profile = configured
    path.precomputed_lane_reference = profile
    path.target_lane_idx = 0
    path.is_overtaking = True
    mpc.set_soft_lateral_reference()
    mpc.set_lane_transition_weights()
    wp = path.get_waypoint(wp_id)
    row = profile.sample(wp_id, 0)
    mpc.model.update_states(wp.x-row[0]*np.sin(wp.psi), wp.y+row[0]*np.cos(wp.psi), wp.psi+row[1])
    mpc.model.wp_id = wp_id
    mpc.model.spatial_state = mpc.model.t2s(wp, mpc.model.temporal_state)
    with redirect_stdout(io.StringIO()):
        mpc._init_problem(mpc.N, 0.)
        result = mpc.optimizer.solve()
    assert result.info.status == 'solved'
