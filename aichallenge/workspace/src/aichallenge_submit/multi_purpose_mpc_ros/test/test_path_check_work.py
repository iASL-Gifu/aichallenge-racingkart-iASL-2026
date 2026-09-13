"""Exact reuse and conservative broad-phase equivalence."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
import math
import numpy as np
import pytest
from multi_purpose_mpc_ros import collision_geometry as cg
from multi_purpose_mpc_ros.core import path_check_work as work
from multi_purpose_mpc_ros.core.runtime_diagnostics import RuntimeDiagnostics


def controller():
    diag = RuntimeDiagnostics(.025); diag.begin(0.)
    grid = NS(data_backup=np.ones((8,8)), resolution=.1, origin=[0.,0.],
              width=8, height=8, static_recovery_path_is_clear=Mock(return_value=(True,'clear')))
    return NS(_map=grid, _path_check_work=work.PathCheckWork(diag))


def test_wall_cache_exact_keys_modes_map_and_cycle():
    c=controller();g=cg.BodyGeometry();a=cg.BodyPose(0.,0.,0.,1.)
    assert work.wall_clear(c,[a],g)==(True,'clear')
    work.wall_clear(c,[a],g)
    assert c._map.static_recovery_path_is_clear.call_count==1
    for body in [replace(a,x=1e-10),replace(a,yaw=.1),replace(a,stamp=2.),
                 replace(a,position_valid=False),replace(a,uncertainty=.1)]:
        work.wall_clear(c,[body],g)
    work.wall_clear(c,[a],g,allowance=.03)
    work.wall_clear(c,[a],g,slide=True)
    work.wall_clear(c,[a],replace(g,width=1.6))
    c._map.data_backup=c._map.data_backup.copy();work.wall_clear(c,[a],g)
    c._path_check_work=work.PathCheckWork();work.wall_clear(c,[a],g)
    assert c._map.static_recovery_path_is_clear.call_count==11


@pytest.mark.parametrize('safe',[False,True])
def test_traffic_cache_invalidation_and_separation_preserved(monkeypatch,safe):
    c=controller();g=cg.BodyGeometry();a=cg.BodyPose(0.,0.,0.,0.)
    bodies=[a,replace(a,x=1.)];target=replace(a,x=4.)
    sweep=Mock(return_value=False);separate=Mock(return_value=safe)
    monkeypatch.setattr(cg,'swept_path_clear',sweep);monkeypatch.setattr(cg,'separating_path_clear',separate)
    assert work.traffic_clear(c,bodies,[0.,1.],target,(0.,0.),g)==safe
    assert work.traffic_clear(c,bodies,[0.,1.],target,(0.,0.),g)==safe
    assert sweep.call_count==separate.call_count==1
    variants=[(bodies,[0.,1.01],target,(0.,0.),g,False),
              (bodies,[0.,1.],replace(target,y=.01),(0.,0.),g,False),
              (bodies,[0.,1.],replace(target,stamp=.01),(0.,0.),g,False),
              (bodies,[0.,1.],target,(.01,0.),g,False),
              (bodies,[0.,1.],target,(0.,0.),g,True),
              ([a,replace(a,y=.01)],[0.,1.],target,(0.,0.),g,False)]
    for ps,ts,t,v,geo,rev in variants:
        assert work.traffic_clear(c,ps,ts,t,v,geo,reverse=rev)==safe
    assert sweep.call_count==separate.call_count==7
    c._path_check_work=work.PathCheckWork()
    work.traffic_clear(c,bodies,[0.,1.],target,(0.,0.),g)
    assert sweep.call_count==8


def test_broad_phase_matches_original_sweeps(monkeypatch):
    rng=np.random.default_rng(181112);g=cg.BodyGeometry()
    optimized=cg._segment_definitely_separated
    skipped=0
    for i in range(300):
        a=cg.BodyPose(*rng.uniform(-3,3,3),0.,uncertainty=float(rng.uniform(0,.5)))
        b=replace(a,x=a.x+rng.uniform(-2,2),y=a.y+rng.uniform(-2,2),
                  yaw=a.yaw+rng.uniform(-1,1),lateral_uncertainty=float(rng.uniform(0,.5)))
        t=cg.BodyPose(*rng.uniform(-15,15,3),0.,uncertainty=float(rng.uniform(0,.7)))
        if i%7==0:t=replace(t,yaw=None)
        velocity=tuple(rng.uniform(-15,15,2));times=[0.,float(rng.uniform(.01,2))]
        monkeypatch.setattr(cg,'_segment_definitely_separated',optimized)
        actual=cg.swept_path_clear([a,b],times,t,velocity,g)
        monkeypatch.setattr(cg,'_segment_definitely_separated',lambda *args:False)
        expected=cg.swept_path_clear([a,b],times,t,velocity,g)
        assert actual==expected
    monkeypatch.setattr(cg,'_segment_definitely_separated',optimized)
    a=cg.BodyPose(0.,0.,0.,0.);b=replace(a,x=.1)
    # A currently distant car crossing the path is still rejected.
    assert not cg.swept_path_clear([a,b],[0.,1.],replace(a,y=10.),(0.,-20.),g)
    samples=Mock(wraps=cg._sweep_ego_samples)
    monkeypatch.setattr(cg,'_sweep_ego_samples',samples)
    assert cg.swept_path_clear([a,b],[0.,1.],replace(a,y=30.),(0.,0.),g)
    samples.assert_not_called()
    assert not cg.swept_path_clear([a,b],[0.,1.],replace(a,position_valid=False),(0.,0.),g)


def test_wall_fast_path_matches_full_footprint_evidence():
    import inspect
    import textwrap
    from multi_purpose_mpc_ros.core.map import Map
    from .test_static_reverse_clearance import make_map
    # Reference executes the same full oriented footprint test without early exit.
    source=textwrap.dedent(inspect.getsource(Map._compute_static_body_collision_detail))
    source=source.replace('    if not occupied_region.any():\n        return None\n','')
    scope=dict(Map._compute_static_body_collision_detail.__globals__)
    exec(source,scope)
    reference=scope['_compute_static_body_collision_detail']
    grid=make_map(width=120,height=120,resolution=.1)
    grid.data_backup[:,35]=0;grid.data_backup[70,20:100]=0
    rng=np.random.default_rng(230);g=cg.BodyGeometry()
    for _ in range(300):
        a=cg.BodyPose(float(rng.uniform(-1,13)),float(rng.uniform(-1,13)),
                      float(rng.uniform(-math.pi,math.pi)),0.,
                      uncertainty=float(rng.uniform(0,.2)))
        for cells in (False,True):
            assert grid._compute_static_body_collision_detail(a,g,.05,include_cells=cells)==reference(grid,a,g,.05,include_cells=cells)


def test_body_preparation_reuses_only_exact_origin_pose_and_observation(monkeypatch):
    c=controller();path=[(0.,0.,0.),(1.,0.,.1)]
    build=Mock(wraps=cg.ego_body);monkeypatch.setattr(cg,'ego_body',build)
    first=work.prepare_bodies(c,path)
    assert work.prepare_bodies(c,path) is first
    assert build.call_count==2
    changes=[('_collision_now',1.),('_collision_ego_metadata',(1.,'map',False)),
             ('_collision_ego_alignment',(.1,.2,.3,1.)),
             ('_collision_ego_origin','rear_axle'),('_collision_center_offset',.7),
             ('_collision_origin_lateral_margin',.1)]
    for key,value in changes:
        setattr(c,key,value);work.prepare_bodies(c,path)
    work.prepare_bodies(c,[(0.,0.,0.),(1.,0.,.100001)])
    assert build.call_count==16


def test_interpolation_cache_values_pose_times_delay_and_cycle(monkeypatch):
    from multi_purpose_mpc_ros.core import control_continuity
    c=controller();pose=NS(x=0.,y=0.,theta=0.)
    mpc=NS(current_recovery_prediction=np.array([[0.,0.,0.],[1.,.1,.2]]),
           current_prediction_times=np.array([0.,1.]))
    compute=Mock(wraps=control_continuity.timed_mpc_path)
    monkeypatch.setattr(control_continuity,'timed_mpc_path',compute)
    first=work.prepare_mpc_path(c,mpc,pose,.15)
    assert work.prepare_mpc_path(c,mpc,pose,.15) is first
    assert compute.call_count==1
    mpc.current_recovery_prediction[1,0]+=1e-9
    assert work.prepare_mpc_path(c,mpc,pose,.15)!=first
    mpc.current_prediction_times[1]=2.
    work.prepare_mpc_path(c,mpc,pose,.15)
    pose.theta=.1;work.prepare_mpc_path(c,mpc,pose,.15)
    work.prepare_mpc_path(c,mpc,pose,.16)
    c._path_check_work=work.PathCheckWork()
    work.prepare_mpc_path(c,mpc,pose,.16)
    assert compute.call_count==6


def test_wall_margin_configuration_changes_invalidate_verdict():
    c=controller();c._cfg=NS(mpc=NS(prediction_outer_boundary_guard=.1))
    body=cg.BodyPose(0.,0.,0.,0.);g=cg.BodyGeometry()
    work.wall_clear(c,(body,),g)
    assert c._map.static_recovery_path_is_clear.call_args.kwargs['wall_margin']==.1
    c._cfg.mpc.prediction_outer_boundary_guard=.2
    work.wall_clear(c,(body,),g)
    assert c._map.static_recovery_path_is_clear.call_count==2
    assert c._map.static_recovery_path_is_clear.call_args.kwargs['wall_margin']==.2
