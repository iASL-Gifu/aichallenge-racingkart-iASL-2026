import numpy as np
from scipy import sparse

from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.fuzzy_weight_adapter import FuzzyWeightAdapter


class _DummyReferencePath:
    def update_simple_path_constraints(self, _horizon, _margin):
        pass


class _DummyModel:
    n_states = 3
    Ts = 0.025
    length = 1.087
    safety_margin = 0.0
    reference_path = _DummyReferencePath()


def _adapter(**overrides):
    arguments = dict(
        lateral_error_full_scale=1.5,
        heading_error_full_scale_rad=np.deg2rad(15.0),
        q_lateral_min_ratio=0.30,
        q_heading_min_ratio=0.40,
        steer_delta_min_ratio=0.25,
        smoothing_sec=0.0,
        control_period_sec=0.025,
    )
    arguments.update(overrides)
    return FuzzyWeightAdapter(**arguments)


def _mpc():
    return MPC(
        _DummyModel(),
        5,
        sparse.diags([3.0e6, 5.5e7, 2.0e6]),
        sparse.diags([1.0e5, 1.0e3]),
        sparse.diags([1.0e6, 1.0e6, 1.0e4]),
        {"xmin": np.full(3, -np.inf), "xmax": np.full(3, np.inf)},
        {"umin": np.asarray([0.0, -1.0]), "umax": np.asarray([12.0, 1.0])},
        18.0,
        3.0,
        0,
        False,
        False,
    )


def test_zero_error_prioritizes_lateral_tracking():
    ratios = _adapter().update(0.0, 0.0)
    assert ratios.q_lateral > 0.80
    assert ratios.q_heading >= 0.40


def test_large_lateral_error_increases_steering_smoothness_weight():
    adapter = _adapter()
    centered = adapter.update(0.0, 0.0)
    far = adapter.update(1.5, 0.0)
    assert far.steer_delta > centered.steer_delta
    assert far.q_lateral < centered.q_lateral


def test_weight_floors_are_enforced():
    ratios = _adapter(
        q_lateral_min_ratio=0.6,
        q_heading_min_ratio=0.7,
        steer_delta_min_ratio=0.8,
    ).update(1.5, -np.deg2rad(15.0))
    assert ratios.q_lateral >= 0.6
    assert ratios.q_heading >= 0.7
    assert ratios.steer_delta >= 0.8


def test_dynamic_cost_update_preserves_sparse_structure():
    mpc = _mpc()
    mpc.configure_fuzzy_weights(
        enabled=True,
        lateral_error_full_scale=1.5,
        heading_error_full_scale_rad=np.deg2rad(15.0),
        q_lateral_min_ratio=0.30,
        q_heading_min_ratio=0.40,
        steer_delta_min_ratio=0.25,
        steer_delta_max_weight=1000.0,
        smoothing_sec=0.15,
    )
    indices = mpc.P_base.indices.copy()
    indptr = mpc.P_base.indptr.copy()
    data_before = mpc.P_base.data.copy()

    mpc._apply_cost_matrix_values(0.35, 0.55, 0.75)

    np.testing.assert_array_equal(mpc.P_base.indices, indices)
    np.testing.assert_array_equal(mpc.P_base.indptr, indptr)
    assert len(mpc.P_base.data) == len(data_before)
    assert not np.array_equal(mpc.P_base.data, data_before)


def test_disabled_fuzzy_cost_matches_original_block_diagonal():
    mpc = _mpc()
    expected = sparse.block_diag([
        sparse.kron(sparse.eye(mpc.N), mpc.Q),
        mpc.QN,
        sparse.kron(sparse.eye(mpc.N), mpc.R),
    ], format="csc")
    np.testing.assert_allclose(mpc.P_base.toarray(), expected.toarray())
