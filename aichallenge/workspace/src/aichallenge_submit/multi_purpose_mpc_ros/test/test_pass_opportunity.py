import math
from types import SimpleNamespace as NS
import pytest
from multi_purpose_mpc_ros.core.pass_opportunity import evaluate,controller_opportunity


def test_matched_normal_car_is_not_a_pass():
    assert not evaluate(28.,9.,35/3.6).allowed


def test_clear_advantage_can_pass():
    assert evaluate(12.,5.,35/3.6).allowed


def test_far_car_cannot_be_caught_in_time():
    assert not evaluate(35.,6.,35/3.6).allowed


@pytest.mark.parametrize('speed',[0.,.3,1.5])
def test_stopped_ultra_slow_exception(speed):
    assert evaluate(35.,speed,2.).allowed


@pytest.mark.parametrize('speed',[math.nan,math.inf])
def test_unknown_speeds_block(speed):
    assert not evaluate(20.,speed,9.).allowed


def test_not_enough_road_even_with_time():
    assert not evaluate(12.,5.,9.7,available_distance=20.).allowed


def test_reference_speed_not_follow_command():
    def waypoint(i, speed=9.7):
        angle=2*math.pi*(i%100)/100
        return NS(v_ref=speed,x=20*math.cos(angle),y=20*math.sin(angle))
    path=NS(n_waypoints=100,circular=True,
            get_waypoint=waypoint)
    c=NS(_reference_pathN_center=path,_mpcN_center=NS(input_constraints={'umax':[9.7,1.]}),
         _ultra_slow_early_commit_speed=1.5,_pass_opportunity_distance=80.,
         _pass_opportunity_max_seconds=8.,_pass_opportunity_min_gain=1.,
         _pass_opportunity_speed_reserve=.5,_pass_opportunity_clearance=3.,
         _last_u=[5.,0.])
    assert controller_opportunity(c,90,12.,5.,True).allowed
    path.get_waypoint=lambda i:waypoint(i,5.5 if i%100==10 else 9.7)
    assert not controller_opportunity(c,90,12.,5.,True).allowed
    path.circular=False
    path.get_waypoint=lambda i:NS(v_ref=9.7,x=float(i),y=0.)
    assert not controller_opportunity(c,99,12.,5.,True).allowed


def test_gate_is_before_latch_and_keeps_existing_manoeuvres():
    # Controller source is too large to run a ROS control cycle in this unit
    # suite; assert the admission predicate structurally, rather than logging.
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'multi_purpose_mpc_ros/mpc_controller.py').read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and '_pass_opportunity_enabled' in ast.unparse(n.test)]
    assert len(nodes)==1
    predicate=compile(ast.Expression(nodes[0].test),'<gate>','eval')
    values=dict(self=NS(_pass_opportunity_enabled=True,_ultra_slow_early_commit_speed=1.5),
                existing_outer_latch=False,latch_candidate_lane_idx=0,lead_is_stationary=False,
                opponent_velocity_valid=True,opponent_v_lead=7.)
    assert eval(predicate,values)
    for changes in [dict(existing_outer_latch=True),dict(opponent_v_lead=1.),dict(latch_candidate_lane_idx=1),dict(lead_is_stationary=True)]:
        assert not eval(predicate,{**values,**changes})


def test_acceleration_delay_blocks_log_like_early_pass():
    args=dict(distance=16.5, lead_speed=6.75, available_speed=9.72, min_gain=2.)
    assert evaluate(**args).allowed
    assert not evaluate(**args,current_speed=7.48,acceleration=1.).allowed
    assert evaluate(**{**args,'distance':10.},current_speed=7.48,acceleration=1.).allowed


def test_matched_following_can_accelerate_to_pass():
    assert evaluate(10.,5.,9.72,current_speed=5.,closing_speed=0.,acceleration=1.).allowed


def test_observed_opening_gap_makes_estimate_conservative():
    assert evaluate(10.,5.,9.72,current_speed=7.,acceleration=1.).allowed
    assert not evaluate(10.,5.,9.72,current_speed=7.,closing_speed=-1.,acceleration=1.).allowed


def test_closing_history_is_target_freshness_and_clock_scoped():
    from multi_purpose_mpc_ros.core.pass_opportunity import observe_closing
    c=NS()
    assert observe_closing(c,'a',1.,20.,True) is None
    assert observe_closing(c,'a',1.4,19.6,True)==pytest.approx(1.)
    assert observe_closing(c,'b',1.5,10.,True) is None
    assert observe_closing(c,'b',1.9,10.,True)==0.
    assert observe_closing(c,'b',1.8,10.,True) is None
    assert observe_closing(c,'b',3.,10.,True) is None
    assert observe_closing(c,'b',3.4,30.,True) is None
    assert observe_closing(c,'b',3.5,30.,False) is None


@pytest.mark.parametrize('speed',[0.,.3,1.5])
def test_ultra_slow_is_not_delayed_by_motion_estimate(speed):
    assert evaluate(35.,speed,2.,current_speed=0.,closing_speed=-2.).allowed
