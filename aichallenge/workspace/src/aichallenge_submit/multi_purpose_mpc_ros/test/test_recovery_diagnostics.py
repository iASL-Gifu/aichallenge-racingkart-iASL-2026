import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from multi_purpose_mpc_ros.core.recovery_diagnostics import log_event


def controller():
    m=NS(model=NS(wp_id=3, temporal_state=NS(x=0.,y=0.,psi=0.),
                  reference_path=NS(target_lane_idx=None)),failure_reason=None,
         _prediction_lower_bounds=np.array([0.,0.]),
         _prediction_upper_bounds=np.array([2.,1.]),
         _constraint_wp_ids=np.array([9,10]),
         _constraint_physical_free_widths=np.array([3.6,2.6]))
    logger=Mock()
    return NS(_mpc=m,_overtake=NS(requested_lane=None,target_id='d3'),
              _mpc_safety_recovery_active=True,get_logger=lambda:logger,
              get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_000_000_000)))


def test_release_and_reentry_keep_separate_snapshots_and_widths():
    c=controller()
    log_event(c,'release',command=[1.,0.])
    c._mpc._prediction_upper_bounds[1]=0.
    c._mpc.failure_reason='collapsed'
    log_event(c,'enter')
    data=json.loads(c.get_logger().info.call_args.args[0].split('] ',1)[1])
    assert data['bounds'][0]['wp']==10
    assert data['bounds'][0]['reference_ub']==0.
    assert data['bounds'][0]['physical_width']==2.6
    assert data['previous_release']['bounds'][0]['reference_ub']==1.
    assert data['previous_release']['failure'] is None


def test_rejected_path_throttled_and_nonfinite_is_json_null():
    c=controller()
    log_event(c,'path_rejected',reason='vehicle_collision=d2',time=float('nan'))
    log_event(c,'path_rejected')
    assert c.get_logger().info.call_count==1
    data=json.loads(c.get_logger().info.call_args.args[0].split('] ',1)[1])
    assert data['extra']['time'] is None


def test_diagnostic_failure_does_not_break_control():
    c=controller()
    c._mpc=None
    log_event(c,'enter')
    c.get_logger().warn.assert_called_once()


def test_lazy_payload_not_built_on_suppressed_ticks():
    c=controller()
    payload=Mock(return_value={'command':[1.,0.]})
    log_event(c,'speed_limit',extra_factory=payload)
    log_event(c,'speed_limit',extra_factory=payload)
    payload.assert_called_once()
    # State transitions are not subject to the recurring-event throttle.
    log_event(c,'release',extra_factory=payload)
    log_event(c,'enter',extra_factory=payload)
    assert payload.call_count == 3


def test_repeated_event_reports_summary_without_collecting_snapshot(monkeypatch):
    from multi_purpose_mpc_ros.core import recovery_diagnostics as module
    c=controller()
    seconds=[10.]
    c.get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(seconds[0]*1e9)))
    collect=Mock(wraps=module.snapshot)
    monkeypatch.setattr(module,'snapshot',collect)
    log_event(c,'path_rejected',reason='wall')
    seconds[0]=11.
    log_event(c,'path_rejected',reason='wall')
    assert collect.call_count == 1
    data=json.loads(c.get_logger().info.call_args.args[0].split('] ',1)[1])
    assert data['summary'] and 'bounds' not in data
    seconds[0]=12.
    log_event(c,'path_rejected',reason='vehicle_collision=d2')
    assert collect.call_count == 2


def test_detail_gate_checks_before_collection_and_allows_transitions_and_rollback():
    from multi_purpose_mpc_ros.core.recovery_diagnostics import diagnostic_due
    c=controller()
    seconds=[10.]
    c.get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(seconds[0]*1e9)))
    assert diagnostic_due(c,'lane',(0,'failed'))
    assert not diagnostic_due(c,'lane',(0,'failed'))
    assert diagnostic_due(c,'lane',(2,'failed'))
    seconds[0]=11.
    assert diagnostic_due(c,'lane',(2,'failed'))
    seconds[0]=1.
    assert diagnostic_due(c,'lane',(2,'failed'))
