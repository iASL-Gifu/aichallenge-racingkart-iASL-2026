"""Regression tests for diagnostic-only outer commit prediction snapshots."""

import inspect

from multi_purpose_mpc_ros.mpc_controller import MPCController


def test_prediction_diagnostic_interpolates_candidate_at_actual_time():
    samples = [
        {"time": 0.0, "e_y": 0.0},
        {"time": 0.2, "e_y": 2.0},
    ]

    assert MPCController._prediction_diagnostic_interpolate(
        samples, 0.1, "e_y") == 1.0


def test_stale_candidate_gate_snapshot_is_not_attached_to_commit():
    controller = MPCController.__new__(MPCController)
    controller._loop = 11
    controller._candidate_envelope_diagnostic_cache = {
        2: {"control_loop": 10, "samples": []},
    }
    controller._outer_commit_prediction_diagnostic = None

    controller._start_outer_commit_prediction_diagnostic(
        source="normal", vehicle_id="d3", lane_idx=2,
        pose=None, speed=1.0, now_sec=2.0)

    assert controller._outer_commit_prediction_diagnostic is None


def test_actual_capture_rejects_non_committed_lane_ownership():
    controller = MPCController.__new__(MPCController)
    controller._outer_commit_prediction_diagnostic = {
        "lane_idx": 2,
        "commit_time": 1.0,
        "actual_samples": None,
    }
    controller._target_lane_idx = 0

    controller._capture_outer_commit_actual_prediction_diagnostic(
        applied_lane_idx=0, transition_duration_sec=0.6, now_sec=1.1)

    assert controller._outer_commit_prediction_diagnostic[
        "actual_samples"] is None


def test_all_three_new_outer_commit_sources_start_diagnostics():
    control_source = inspect.getsource(MPCController._control)

    assert '"preempt" if l1_rejoin_preempt_commit else "normal"' in control_source
    assert control_source.count('source="fallback"') >= 2
    assert "_capture_outer_commit_actual_prediction_diagnostic" in control_source

