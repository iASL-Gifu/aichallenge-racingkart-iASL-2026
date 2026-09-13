from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from .test_overtake_session import controller_method as extract_method
from multi_purpose_mpc_ros.core.MPC import lateral_reference_ramp_duration

def controller_method(name):
    f=extract_method(name)
    f.__globals__['lateral_reference_ramp_duration']=lateral_reference_ramp_duration
    return f


def controller():
    c=NS(_overtake=NS(target_id='d4'),_reference_path=object(),
         _prepass_fallback_recovery_active=True,
         _carN_center=NS(wp_id=353,spatial_state=NS(e_y=-.2)),
         _mpcN_center=NS(N=2,_compute_lane_center=lambda wp,lane:2. if lane==2 else -2.,
                        set_soft_lateral_reference=Mock()),
         _l1_soft_rejoin_ramp_sec=1.5,_race_handoff_max_reference_speed=.7,
         _l2_inward_offset_zones=[],_prepass_soft_dropout_grace_sec=.5,
         _prepass_soft_switch_confirm_sec=.3,get_logger=lambda:Mock())
    c._clear_prepass_soft_guidance=lambda:controller_method('_clear_prepass_soft_guidance')(c)
    c._clear_prepass_soft_guidance()
    return c


def test_pause_preserves_anchor_and_elapsed_not_wall_clock():
    c=controller();f=controller_method('_update_prepass_soft_reference')
    f(c,enabled=True,lane_idx=2,now_sec=1.)
    f(c,enabled=True,lane_idx=2,now_sec=2.)
    alpha=c._mpcN_center.set_soft_lateral_reference.call_args.kwargs['alpha']
    c._carN_center.spatial_state.e_y=-1.8
    f(c,enabled=False,lane_idx=2,now_sec=2.)
    f(c,enabled=False,lane_idx=2,now_sec=3.)
    f(c,enabled=True,lane_idx=2,now_sec=4.)
    assert c._prepass_soft_guidance_start_e_y==-.2
    assert c._mpcN_center.set_soft_lateral_reference.call_args.kwargs['alpha']==pytest.approx(alpha)
    c._overtake.target_id='other'
    f(c,enabled=True,lane_idx=2,now_sec=5.)
    assert c._prepass_soft_guidance_start_e_y==-1.8
    assert c._prepass_soft_guidance_started_at==5.


def test_real_side_change_reanchors_even_when_selector_updated_lane_first():
    c=controller();f=controller_method('_update_prepass_soft_reference')
    f(c,enabled=True,lane_idx=2,now_sec=1.)
    c._prepass_soft_candidate_lane_idx=0;c._carN_center.spatial_state.e_y=.6
    f(c,enabled=True,lane_idx=0,now_sec=2.)
    assert c._prepass_soft_guidance_start_e_y==.6
    assert c._prepass_soft_guidance_started_at==2.


def test_ranking_does_not_replace_clear_l2_but_loss_can():
    c=controller();f=controller_method('_latch_prepass_soft_candidate')
    assert f(c,2,1.)==2
    for t in (1.1,1.5,2.):
        assert f(c,0,t,passage={0:False,2:True},conflicts={})==2
    assert f(c,0,3.,passage={0:True,2:False},conflicts={})==2
    assert f(c,0,3.4,passage={0:True,2:False},conflicts={})==0
