"""Widths after body inset must not be compared to the body width again."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from multi_purpose_mpc_ros.core.reference_path import lane_minimum_free_segment_width
from .test_overtake_session import controller_method


@pytest.mark.parametrize('physical,expected', [
    ([2.7], 'physical_below_required_width=0/1'),
    ([1.2], 'physical_below_required_width=1/1'),
    ([], 'physical_free_width_min=nanm'),
])
def test_log_distinguishes_physical_and_reference_widths(physical, expected):
    method = controller_method('_log_lane_constraint_diagnostics')
    method.__globals__['lane_minimum_free_segment_width'] = lane_minimum_free_segment_width
    wp = NS(x=0., y=0., psi=0., kappa=0.)
    path = NS(get_waypoint=lambda _: wp,
              get_lane_bounds=lambda _: [(2.7, 0.)] * 3, inner_lane_width=1.)
    logger = Mock()
    c = NS(_constraint_diagnostics_enabled=True, _constraint_diagnostics_points=5,
           _reference_path=path, _map=NS(obstacles=[]),
           _cfg=NS(bicycle_model=NS(width=1.6)), get_logger=lambda: logger,
           _mpc=NS(_prediction_upper_bounds=[1.9], _prediction_lower_bounds=[.8],
                   _constraint_wp_ids=[264], _constraint_physical_free_widths=physical,
                   failure_reason='test'))
    method(c, lane_idx=2, failed_wp=250, context='test', reason='test')
    logs = '\n'.join(call.args[0] for call in logger.warn.call_args_list)
    assert 'collection failed' not in logs
    assert expected in logs
    assert 'reference_point_width_min=1.100m' in logs
    assert 'physical_free_width=' in logs
    assert 'likely_cause=effective_width_below_required_width' not in logs
