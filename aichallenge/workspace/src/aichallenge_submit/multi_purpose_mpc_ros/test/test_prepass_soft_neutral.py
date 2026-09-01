"""Regression tests for Prepass neutral full-width objective ownership."""

import inspect
from types import SimpleNamespace

from multi_purpose_mpc_ros.mpc_controller import MPCController


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass


class _FakeCenterMpc:
    N = 2

    def __init__(self):
        self.soft_target_lane_idx = 0
        self.set_calls = []

    def set_soft_lateral_reference(self, lane_idx=None, **_kwargs):
        self.soft_target_lane_idx = lane_idx
        self.set_calls.append(lane_idx)

    def _compute_lane_center(self, _wp, lane_idx):
        return -2.0 if lane_idx == 0 else 2.0


def _controller(*, neutral=True):
    controller = MPCController.__new__(MPCController)
    controller._prepass_soft_neutral_hold_active = neutral
    controller._prepass_soft_candidate_lane_idx = 0
    controller._prepass_soft_pending_lane_idx = 2
    controller._prepass_soft_pending_since = 1.0
    controller._prepass_soft_candidate_last_seen_at = 1.0
    controller._prepass_soft_guidance_started_at = 1.0
    controller._prepass_soft_guidance_start_e_y = -0.5
    controller._prepass_soft_guidance_ramp_sec = 1.0
    controller._prepass_soft_switch_confirm_sec = 0.3
    controller._prepass_soft_dropout_grace_sec = 0.1
    controller._prepass_fallback_recovery_stable_since = 1.0
    controller._overtake_target_vehicle_id = "d3"
    controller._mpcN_center = _FakeCenterMpc()
    controller._carN_center = SimpleNamespace(
        spatial_state=SimpleNamespace(e_y=-0.5), wp_id=10)
    controller._l1_soft_rejoin_ramp_sec = 1.5
    controller._race_handoff_max_reference_speed = 0.5
    controller.get_logger = lambda: _FakeLogger()
    return controller


def test_neutral_enter_blocks_same_cycle_soft_reference_reapply():
    controller = _controller(neutral=False)

    controller._enter_prepass_soft_neutral_hold(
        current_candidate=0,
        next_candidate=2,
        heading_error=0.1,
        lateral_speed=0.9,
        yaw_rate=0.2,
        mpc_stable=True,
    )
    controller._update_prepass_soft_reference(
        enabled=True, lane_idx=0, now_sec=2.0)

    assert controller._prepass_soft_neutral_hold_active
    assert controller._mpcN_center.soft_target_lane_idx is None
    assert controller._prepass_soft_candidate_lane_idx == 0
    assert controller._prepass_soft_pending_lane_idx == 2


def test_neutral_blocks_outer_objective_on_later_cycles_and_candidate_changes():
    controller = _controller()

    controller._update_prepass_soft_reference(
        enabled=True, lane_idx=0, now_sec=2.0)
    controller._prepass_soft_candidate_lane_idx = 2
    controller._update_prepass_soft_reference(
        enabled=True, lane_idx=2, now_sec=2.1)

    assert controller._mpcN_center.soft_target_lane_idx is None
    assert controller._prepass_soft_candidate_lane_idx == 2
    assert controller._prepass_soft_pending_lane_idx == 2


def test_neutral_does_not_stop_candidate_pending_tracking():
    controller = _controller()
    controller._prepass_soft_pending_lane_idx = None
    controller._prepass_soft_pending_since = None

    selected = controller._latch_prepass_soft_candidate(2, 2.0)

    assert selected == 0
    assert controller._prepass_soft_pending_lane_idx == 2
    assert controller._prepass_soft_pending_since == 2.0
    assert controller._prepass_soft_neutral_hold_active


def test_soft_guidance_restarts_only_after_neutral_exit():
    controller = _controller()
    controller._update_prepass_soft_reference(
        enabled=True, lane_idx=2, now_sec=2.0)
    assert controller._mpcN_center.soft_target_lane_idx is None

    controller._prepass_soft_neutral_hold_active = False
    controller._update_prepass_soft_reference(
        enabled=True, lane_idx=2, now_sec=2.1)

    assert controller._mpcN_center.soft_target_lane_idx == 2


def test_control_keeps_neutral_out_of_guidance_and_same_cycle_commit():
    source = inspect.getsource(MPCController._control)

    assert "and not self._prepass_soft_neutral_hold_active" in source
    assert "and not neutral_hold_exited_this_cycle" in source
