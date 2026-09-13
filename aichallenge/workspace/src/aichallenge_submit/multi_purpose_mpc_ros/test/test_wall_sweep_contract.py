"""Continuous corner coverage and distinction between body/margin contact."""
import math
from dataclasses import replace
import numpy as np
from multi_purpose_mpc_ros.collision_geometry import BodyGeometry,BodyPose
from multi_purpose_mpc_ros.core.wall_constraints import swept_sample_padding
from .test_static_reverse_clearance import make_map


def test_swept_padding_covers_random_interpolated_uncertain_body_corners():
    rng=np.random.default_rng(481)
    geometry=BodyGeometry()
    for _ in range(300):
        a=BodyPose(4.,4.,rng.uniform(-math.pi,math.pi),0.,uncertainty=rng.uniform(0.,.5))
        b=replace(a,x=a.x+rng.uniform(-.2,.2),y=a.y+rng.uniform(-.2,.2),
                  yaw=a.yaw+rng.uniform(-.1,.1),uncertainty=rng.uniform(0.,.5))
        padding=swept_sample_padding((a,b),geometry)
        for t in np.linspace(0.,1.,21):
            endpoint=a if t<=.5 else b
            yaw=a.yaw+t*(b.yaw-a.yaw)
            uncertainty=a.uncertainty+t*(b.uncertainty-a.uncertainty)
            for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
                lx=sx*(geometry.length/2+uncertainty);ly=sy*(geometry.width/2+uncertainty)
                x=a.x+t*(b.x-a.x)+math.cos(yaw)*lx-math.sin(yaw)*ly-endpoint.x
                y=a.y+t*(b.y-a.y)+math.sin(yaw)*lx+math.cos(yaw)*ly-endpoint.y
                assert abs(x*math.cos(endpoint.yaw)+y*math.sin(endpoint.yaw)) <= geometry.length/2+endpoint.uncertainty+padding+1e-9
                assert abs(-x*math.sin(endpoint.yaw)+y*math.cos(endpoint.yaw)) <= geometry.width/2+endpoint.uncertainty+padding+1e-9


def test_margin_escape_does_not_authorize_new_physical_contact():
    m=make_map(200,200,.1);m.data_backup[:,100]=0
    g=BodyGeometry(length=1.,width=.5)
    start=BodyPose(9.40,10.,0.,0.) # body clear, margin overlaps the wall
    path=[start,replace(start,x=9.5),replace(start,x=9.3)]
    safe,reason=m.static_recovery_path_is_clear(path,g,wall_margin=.1,temporary_depth_increase=.2)
    assert not safe and 'contact=physical' in reason
    safe,reason=m.static_recovery_path_is_clear([start,replace(start,x=9.3)],g,wall_margin=.1)
    assert safe and reason=='wall_escape'


def test_wall_between_samples_is_rejected_even_if_both_bodies_are_clear():
    m=make_map(200,200,.1);m.data_backup[:,100]=0
    g=BodyGeometry(.1,.1)
    safe,_=m.static_recovery_path_is_clear([BodyPose(9.,10.,0.,0.),BodyPose(11.,10.,0.,0.)],g,wall_margin=0.)
    assert not safe


def test_asymmetric_wall_crossing_cannot_look_like_departure_of_sweep_padding():
    m=make_map(200,200,.1);m.data_backup[:,100]=0
    safe,reason=m.static_recovery_path_is_clear(
        [BodyPose(9.7,10.,0.,0.),BodyPose(11.7,10.,0.,0.)],BodyGeometry(.1,.1),wall_margin=.1)
    assert not safe and 'physical_sweep' in reason
