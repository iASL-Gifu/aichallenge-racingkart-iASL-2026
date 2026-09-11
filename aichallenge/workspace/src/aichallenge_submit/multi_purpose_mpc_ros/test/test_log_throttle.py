from types import SimpleNamespace as NS
from unittest.mock import Mock

from multi_purpose_mpc_ros.core import log_throttle
from multi_purpose_mpc_ros.v2x_vehicle_tracker import evaluate_lane_width_samples
from .test_traffic_work import passage_controller
from .test_overtake_session import controller_method


def test_gate_is_bounded_and_independent_per_key(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(log_throttle, 'monotonic', lambda: clock[0])
    owner = NS()
    assert log_throttle.periodic_log_due(owner, 'd2', 5.)
    assert not log_throttle.periodic_log_due(owner, 'd2', 5.)
    assert log_throttle.periodic_log_due(owner, 'd3', 5.)
    clock[0] = 5.
    assert log_throttle.periodic_log_due(owner, 'd2', 5.)
    for i in range(100):
        log_throttle.periodic_log_due(owner, i, 5.)
    assert len(owner._periodic_log_stamps) == 64


def test_passage_logging_off_and_suppressed_never_calls_logger(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(log_throttle, 'monotonic', lambda: clock[0])
    c = passage_controller()
    logger = Mock()
    c.get_logger = Mock(return_value=logger)
    call = controller_method('_vehicle_passage')
    call.__globals__['evaluate_lane_width_samples'] = evaluate_lane_width_samples
    pose = NS(x=0., y=0.)
    expected = call(c, 'd2', pose)
    c.get_logger.assert_not_called()
    c._cfg.mpc = NS(passage_diagnostics_enabled=True)
    c._traffic_work.begin_cycle()
    assert call(c, 'd2', pose) == expected
    assert logger.info.call_count == 2
    clock[0] = 1.
    c._traffic_work.begin_cycle()
    c._reference_pathN_center.waypoints[0].lb = -.5
    assert not call(c, 'd2', pose)[0][0]  # safety updated while logs suppressed
    assert logger.info.call_count == 2
    clock[0] = 5.
    c._traffic_work.begin_cycle()
    assert not call(c, 'd2', pose)[0][0]
    assert logger.info.call_count == 4
    assert all(not entry.kwargs for entry in logger.info.call_args_list)
