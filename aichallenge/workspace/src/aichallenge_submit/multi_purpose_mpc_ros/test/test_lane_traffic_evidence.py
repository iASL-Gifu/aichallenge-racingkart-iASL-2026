from types import SimpleNamespace as NS
from dataclasses import replace
import numpy as np
import pytest
from multi_purpose_mpc_ros import collision_geometry as cg
from multi_purpose_mpc_ros.core import lane_traffic_evidence as evidence
from multi_purpose_mpc_ros.core.curve_priority import select, transition_owner
from .test_overtake_session import controller_method


def fixture(monkeypatch):
    pose=NS(x=0.,y=0.,theta=0.)
    c=NS(_reference_path=object(),_collision_now=10.,_collision_ego_origin='center',
         _v2x_tracker=NS(has_velocity_estimate=lambda v:True,velocity=lambda v:(0.,0.)))
    m=NS(infeasibility_counter=0,current_prediction=object(),used_prediction_fallback=False,
         recovery_requested=False,time_budget_exceeded=False,last_solution_accurate=True,
         current_recovery_prediction=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]),
         current_prediction_times=np.array([0.,.5,1.,1.5]))
    target=cg.BodyPose(1.,5.,0.,10.)
    monkeypatch.setattr(cg,'target_body',lambda *args:target)
    evidence.remember(c,m,0,'lead')
    return c,m,pose,target


def test_unknown_lane_clear_path_and_crossing_target(monkeypatch):
    c,m,p,t=fixture(monkeypatch)
    assert evidence.unknown_lane_conflict(c,p,0,'lead','other') is False
    c._v2x_tracker.velocity=lambda v:(0.,-5.)
    assert evidence.unknown_lane_conflict(c,p,0,'lead','other') is True
    monkeypatch.setattr(cg,'target_body',lambda *args:replace(t,x=.5,y=0.))
    c._v2x_tracker.velocity=lambda v:(0.,0.)
    assert evidence.unknown_lane_conflict(c,p,0,'lead','other') is True


@pytest.mark.parametrize('change',['age','rewind','lane','target','path','pose','short','invalid_body','unknown_yaw','velocity','failed'])
def test_missing_or_incompatible_evidence_never_clears_unknown(monkeypatch,change):
    c,m,p,t=fixture(monkeypatch);lane=0;target='lead'
    if change=='age':c._collision_now+=.251
    if change=='rewind':c._collision_now-=.01
    if change=='lane':lane=2
    if change=='target':target='new'
    if change=='path':c._reference_path=object()
    if change=='pose':p.y=3.
    if change=='short':m.current_prediction_times*=.3;evidence.remember(c,m,0,'lead')
    if change=='invalid_body':monkeypatch.setattr(cg,'target_body',lambda *args:replace(t,position_valid=False))
    if change=='unknown_yaw':monkeypatch.setattr(cg,'target_body',lambda *args:replace(t,yaw=None))
    if change=='velocity':c._v2x_tracker.has_velocity_estimate=lambda v:False
    if change=='failed':evidence.remember(c,m,0,'lead',accepted=False)
    assert evidence.unknown_lane_conflict(c,p,lane,target,'other') is None


def test_snapshot_does_not_alias_live_arrays_and_arrival_times_age(monkeypatch):
    c,m,p,t=fixture(monkeypatch)
    m.current_recovery_prediction[:]=100
    c._collision_now+=.1
    seen=[]
    monkeypatch.setattr(evidence,'traffic_clear',lambda c,b,ts,*args:seen.append(ts) or True)
    assert evidence.unknown_lane_conflict(c,p,0,'lead','other') is False
    assert seen[0][0]==pytest.approx(.1)
    assert seen[0][-1]==pytest.approx(1.5)


@pytest.mark.parametrize('lane',[0,2])
def test_priority_cannot_reverse_admitted_hybrid_but_safety_can_release(lane):
    path=object();hybrid=NS(lane_idx=lane,vehicle_id='lead',length=19.,completed=False)
    c=NS(_reference_path=path,_committed_corridor=NS(path=path,lane=lane),
         _overtake=NS(target_id='lead',hybrid=hybrid),
         _l2_restricted_slow_override=lambda v:False,_lane_horizon_has_vehicle_width=lambda l:True)
    assert transition_owner(c)==lane
    assert select(c,2-lane,2-lane,'lead',{0:True,2:True},{0:{},2:{}})==lane
    hybrid.completed=True
    assert transition_owner(c) is None
    assert select(c,2-lane,2-lane,'lead',{0:True,2:True},{0:{},2:{}})==2-lane
    hybrid.completed=False;c._mpc_safety_recovery_active=True
    assert transition_owner(c) is None


def test_unknown_lane_actual_hold_uses_physical_evidence(monkeypatch):
    from .test_overtake_lane_hold import controller, evidence as hold_evidence
    from .test_v2x_vehicle_tracker import _msg
    c=controller();c._reference_path=object();c._lane_index_for_position=lambda *a:None
    c._v2x_tracker.update(_msg(0., [('car',-10.,2.)]))
    # Integration: a proven conflict remains unsafe; absent proof remains unknown.
    monkeypatch.setattr(evidence,'unknown_lane_conflict',lambda *a:False)
    assert hold_evidence(c)==((),())
    monkeypatch.setattr(evidence,'unknown_lane_conflict',lambda *a:True)
    assert hold_evidence(c)==(('car',),())
    monkeypatch.setattr(evidence,'unknown_lane_conflict',lambda *a:None)
    assert hold_evidence(c)==((),('car',))


def test_logged_wp334_priority_entry_preserves_l2_without_resetting_hybrid():
    path=object();hybrid=NS(lane_idx=2,vehicle_id='d2',length=19.22,completed=False)
    c=NS(_reference_path=path,_committed_corridor=NS(path=path,lane=2),
         _overtake=NS(target_id='d2',hybrid=hybrid),
         get_logger=lambda:NS(info=lambda *a,**kw:None))
    call=controller_method('_apply_l2_restricted_zone_policy')
    assert call(c,2,target_vehicle_id='d2',physical_passage={0:True,2:True},
                conflicts_by_lane={0:{},2:{}},center_wp=334)==2
    assert c._overtake.hybrid is hybrid
    assert hybrid.length==19.22


def test_inaccurate_solution_is_not_unknown_lane_release_evidence(monkeypatch):
    c,m,p,t=fixture(monkeypatch)
    m.last_solution_accurate=False
    evidence.remember(c,m,0,'lead')
    assert evidence.unknown_lane_conflict(c,p,0,'lead','other') is None
