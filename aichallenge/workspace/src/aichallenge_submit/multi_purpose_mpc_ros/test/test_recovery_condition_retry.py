import math
from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts, rollout, evaluate


def failed_attempt(length=.1, initial=.314):
    a=RecoveryAttempts();p=rollout((0,0,0),1,.314,length,1.087)
    a.executing(1,.314,p,initial)
    a.observe(0.,(0,0,0),0.,1,.314,True,commanded_speed=.092)
    a.observe(1.6,(0,0,0),0.,1,.314,True,commanded_speed=.092)
    assert (1,1) in a.excluded
    return a


def test_longer_safe_left_path_can_retry_but_same_failure_cannot():
    a=failed_attempt();p=rollout((0,0,0),1,.314,2.,1.087)
    assert a.retry_is_improved(1,.314,p,.314)
    a.executing(1,.314,p,.314)
    a.observe(2.,(0,0,0),0.,1,.314,True)
    a.observe(3.6,(0,0,0),0.,1,.314,True)
    assert not a.retry_is_improved(1,.314,p,.314)
    assert not a.retry_is_improved(1,.314,rollout((0,0,0),1,.314,.1,1.087),.314)
    a.observe(100.,(0,0,0),0.,1,.314,False)
    assert not a.retry_is_improved(1,.314,p,.314)


def test_preparation_improvement_allows_only_one_attempt_at_same_length():
    a=failed_attempt(2.,-.314);p=rollout((0,0,0),1,.314,2.,1.087)
    assert not a.retry_is_improved(1,.314,p,-.314)
    assert a.retry_is_improved(1,.314,p,.314)
    a.executing(1,.314,p,.314)
    a.observe(2.,(0,0,0),0.,1,.314,True)
    a.observe(3.6,(0,0,0),0.,1,.314,True)
    assert not a.retry_is_improved(1,.314,p,.314)


def select(a,clear):
    return evaluate((0,0,0),target=(3,2),distance=2.,wheelbase=1.087,
        steering_limit=.314,clear=clear,excluded=a.excluded,
        retry_clear=lambda d,s,p:a.retry_is_improved(d,s,p,s))


def test_retry_does_not_bypass_traffic_or_wall_check():
    a=failed_attempt()
    def blocked(path):
        return False,'vehicle_collision=d2'
    motion,_=select(a,blocked)
    assert motion is None
    assert (1,1) in a.excluded


def test_new_candidate_is_checked_and_not_unblocked_until_executed():
    a=failed_attempt();checked=[]
    def clear(path):
        checked.append(path);return True,'clear'
    motion,_=select(a,clear)
    assert motion.steering > 0 and checked
    assert (1,1) in a.excluded
    a.executing(1,motion.steering,motion.poses,motion.steering)
    assert (1,1) not in a.excluded


def test_unknown_failure_conditions_are_not_retried():
    a=RecoveryAttempts();a.excluded.add((1,1))
    assert not a.retry_is_improved(1,.314,rollout((0,0,0),1,.314,2.,1.087),.314)
