import pytest
from multi_purpose_mpc_ros.core.final_emergency_limit import enforce_emergency_limit


def apply(speed=8., acc=2., boost=True, **overrides):
    kw=dict(limit=4., measured_speed=6., kp=2., a_min=-8., a_max=3., active=True)
    kw.update(overrides)
    return enforce_emergency_limit(speed, acc, boost, **kw)


def test_no_new_hazard_no_intervention():
    assert apply(active=False) == (8.,2.,True)
    assert apply(limit=None) == (8.,2.,True)


def test_existing_cap_and_stronger_brake_never_raised():
    assert apply() == (4.,-4.,False)
    assert apply(speed=2.,acc=-8.,boost=False) == (2.,-8.,False)
    assert apply(speed=4.,acc=-4.,boost=False) == (4.,-4.,False)


def test_stop_cap_and_reverse():
    assert apply(limit=0.) == (0.,-8.,False)
    assert apply(speed=-1.) == (-1.,2.,True)


def test_final_guard_is_after_smoothing_before_saved_and_published_command():
    from .test_hybrid_integration import control_tree
    import ast
    source=ast.unparse(control_tree())
    assert source.index('limited = enforce_emergency_limit') < source.index('self._last_acc = acc')
    assert source.index('self._last_acc = acc') < source.index('self._publish_control_command')
