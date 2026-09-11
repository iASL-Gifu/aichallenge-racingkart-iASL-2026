import math
from types import SimpleNamespace as NS
import pytest
from multi_purpose_mpc_ros import collision_geometry as cg


@pytest.mark.parametrize("margin", [0.0, 0.272])
def test_unknown_origin_circle_reduced_but_known_yaw_rectangle_unchanged(margin):
    options = dict(origin='unconfirmed', offset=.522, uncertainty=.01, origin_lateral_margin=.272)
    old = cg.body_pose(0.,0.,None,0.,**options)
    new = cg.body_pose(0.,0.,None,0.,unknown_yaw_origin_margin=margin,**options)
    geom = cg.BodyGeometry()
    assert old.uncertainty-new.uncertainty == pytest.approx(.522-margin)
    assert cg.extents(new,geom,0.)[0] == pytest.approx(geom.radius+margin+.01)
    assert math.hypot(*cg.outline(new,geom)[0]) == pytest.approx(geom.radius+margin+.01)
    box = cg.body_pose(10.,0.,0.,0.)
    assert cg.overlaps(new,box,geom) is False
    assert cg.overlaps(new,new,geom) is True
    for direction in (False,True):
        assert cg.body_pose(0.,0.,.5,0.,direction_valid=direction,**options) == cg.body_pose(
            0.,0.,.5,0.,direction_valid=direction,unknown_yaw_origin_margin=margin,**options)


def test_margin_only_changes_unconfirmed_origin_and_invalid_config_falls_back():
    assert cg.body_pose(0.,0.,None,0.,origin='center') == cg.body_pose(
        0.,0.,None,0.,origin='center',unknown_yaw_origin_margin=.272)
    old=cg.body_pose(0.,0.,None,0.,origin='unconfirmed')
    assert cg.body_pose(0.,0.,None,0.,origin='unconfirmed',unknown_yaw_origin_margin=float('nan')) == old


def test_controller_target_snapshot_uses_margin_but_ego_keeps_original():
    from .test_collision_body_pose import tracker, update
    t=tracker();update(t,0.,0.)
    c=NS(_v2x_tracker=t.snapshot(),_collision_now=0.,
         _collision_v2x_unknown_yaw_origin_margin=.272,
         _collision_origin_lateral_margin=.272)
    target=cg.target_body(c,'d2')
    assert target.yaw is None
    assert target.uncertainty == pytest.approx(.272)
    ego=cg.ego_body(c,NS(x=0.,y=0.,theta=0.))
    assert ego.uncertainty == pytest.approx(.522)
    assert ego.lateral_padding == pytest.approx(.272)
