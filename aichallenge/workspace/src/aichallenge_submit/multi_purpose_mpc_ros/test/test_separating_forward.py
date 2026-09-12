from multi_purpose_mpc_ros import collision_geometry as cg

G = cg.BodyGeometry()

def path(xs, ys=None, yaws=None):
    return [cg.BodyPose(x, 0. if ys is None else ys[i],
                        0. if yaws is None else yaws[i], 0.) for i,x in enumerate(xs)]

def check(poses, target=None, velocity=(0.,0.)):
    target = target or cg.BodyPose(-G.length-.06, 0., 0., 0.)
    return cg.separating_forward_path_clear(poses, [i*.1 for i in range(len(poses))], target, velocity, G)

def test_separating_margin_overlap_is_allowed_but_strict_sweep_rejects():
    poses=path([i*.02 for i in range(21)])
    target=cg.BodyPose(-G.length-.06,0.,0.,0.)
    assert not cg.swept_path_clear(poses,[i*.1 for i in range(21)],target,(0.,0.),G)
    assert check(poses,target)

def test_real_contact_and_uncertainty_remain_rejected():
    poses=path([i*.02 for i in range(21)])
    assert not check(poses,cg.BodyPose(-G.length+.01,0.,0.,0.))
    assert not check(poses,cg.BodyPose(-G.length-.06,0.,0.,0.,uncertainty=.1))

def test_reverse_and_insufficient_separation_rejected():
    assert not check(path([0.,-.02,-.04]))
    assert not check(path([0.,.01,.02]))

def test_reapproach_by_moving_rear_vehicle_rejected():
    assert not check(path([i*.02 for i in range(21)]),velocity=(.3,0.))

def test_turn_sweeps_rear_corner_into_vehicle():
    poses=path([0.,.01,.02,.03], yaws=[0.,.15,.3,.45])
    assert not check(poses)

def test_unknown_yaw_does_not_receive_exception():
    assert not check(path([0.,.1,.2]),cg.BodyPose(-G.length-.06,0.,None,0.))

def test_clear_initial_pose_does_not_receive_exception():
    assert not check(path([0.,.1,.2]),cg.BodyPose(-G.length-.5,0.,0.,0.))

def test_shared_controller_gate_forward_reverse_and_static_checks():
    from types import SimpleNamespace as NS, MethodType
    from unittest.mock import patch
    from .test_overtake_session import controller_method
    target=cg.BodyPose(-G.length-.06,0.,0.,0.)
    c=NS(_recovery_localization_available=True, _straight_reentry_speed=1.,
         _reverse_overlap_allowance=.03, _collision_now=0., _collision_ego_origin="center",
         _map=NS(static_recovery_path_is_clear=lambda *a,**kw:(True,'clear')),
         _v2x_tracker=NS(active_vehicle_ids=lambda:['d4'],
                         has_velocity_estimate=lambda vid:True,
                         velocity=lambda vid:(0.,0.)))
    c.check=MethodType(controller_method('_reentry_path_is_clear'),c)
    poses=[(i*.02,0.,0.) for i in range(21)]
    times=[i*.1 for i in range(21)]
    with patch.object(cg,'target_body',return_value=target):
        assert c.check(poses,times=times)[0]
        assert not c.check(poses,times=times,reverse=True)[0]
        c._map.static_recovery_path_is_clear=lambda *a,**kw:(False,'wall')
        assert c.check(poses,times=times)==(False,'wall')
