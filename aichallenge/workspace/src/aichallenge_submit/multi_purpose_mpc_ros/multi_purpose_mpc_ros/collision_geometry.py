"""Shared map-frame body geometry, including explicitly unknown headings/origins."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class BodyPose:
    x: float
    y: float
    yaw: float | None
    stamp: float
    frame: str = 'map'
    yaw_source: str = 'unknown'
    direction_valid: bool = False
    position_valid: bool = True
    origin: str = 'center'
    center_offset: float = 0.0
    uncertainty: float = 0.0
    yaw_stamp: float | None = None
    lateral_uncertainty: float | None = None

    @property
    def lateral_padding(self):
        return self.uncertainty if self.lateral_uncertainty is None else self.lateral_uncertainty

    @property
    def yaw_valid(self):
        return self.yaw is not None and math.isfinite(self.yaw)


@dataclass(frozen=True)
class BodyGeometry:
    length: float = 2.064
    width: float = 1.45

    @property
    def radius(self):
        return math.hypot(self.length / 2, self.width / 2)


def body_pose(x, y, yaw, stamp, *, frame='map', source='unknown',
              direction_valid=False, origin='center', offset=0.522, uncertainty=0.0, origin_lateral_margin=None):
    valid = frame == 'map' and all(math.isfinite(v) for v in (x, y, stamp, uncertainty))
    known = yaw is not None and math.isfinite(yaw)
    shift = 0.0
    lateral_uncertainty = uncertainty
    if origin == 'rear_axle' and known and direction_valid:
        shift = offset
        x += shift * math.cos(yaw)
        y += shift * math.sin(yaw)
    elif origin != 'center':
        # No guessed forward/reverse direction or unconfirmed antenna offset.
        uncertainty += abs(offset)
        # Keep longitudinal origin ambiguity; tune only the known body lateral axis.
        lateral_uncertainty += (abs(offset) if origin_lateral_margin is None
                                else min(abs(offset), max(float(origin_lateral_margin), 0.0)))
    return BodyPose(x, y, yaw if known else None, stamp, frame, source,
                    direction_valid, valid, origin, shift, uncertainty, stamp if known else None, lateral_uncertainty)


def extents(body, geometry, heading):
    if not body.yaw_valid:
        r = geometry.radius + body.uncertainty
        return r, r
    angle = body.yaw - heading
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    return ((geometry.length / 2+body.uncertainty)*c + (geometry.width / 2+body.lateral_padding)*s,
            (geometry.length / 2+body.uncertainty)*s + (geometry.width / 2+body.lateral_padding)*c)


def overlaps(first, second, geometry, margin=0.0):
    """True includes possible overlap under uncertainty; None means invalid position."""
    if not first.position_valid or not second.position_valid:
        return None
    geometry = BodyGeometry(geometry.length+max(margin,0.), geometry.width+max(margin,0.))
    pad = 0.0
    if not first.yaw_valid and not second.yaw_valid:
        return math.hypot(second.x-first.x, second.y-first.y) <= (
            2*geometry.radius + first.uncertainty + second.uncertainty + 2*pad)
    if not first.yaw_valid or not second.yaw_valid:
        circle, box = (first, second) if not first.yaw_valid else (second, first)
        dx, dy = circle.x-box.x, circle.y-box.y
        c,s = math.cos(box.yaw), math.sin(box.yaw)
        x,y = abs(dx*c+dy*s), abs(-dx*s+dy*c)
        return math.hypot(max(x-geometry.length/2-box.uncertainty-pad,0.),
                          max(y-geometry.width/2-box.lateral_padding-pad,0.)) <= geometry.radius+circle.uncertainty+pad
    dx,dy = second.x-first.x,second.y-first.y
    for heading in (first.yaw, first.yaw+math.pi/2, second.yaw, second.yaw+math.pi/2):
        distance = abs(dx*math.cos(heading)+dy*math.sin(heading))
        if distance > extents(first,geometry,heading)[0]+extents(second,geometry,heading)[0]+2*pad:
            return False
    return True


def outline(body, geometry):
    if not body.yaw_valid:
        r=geometry.radius+body.uncertainty
        return [(body.x+r*math.cos(i*math.pi/16),body.y+r*math.sin(i*math.pi/16)) for i in range(33)]
    a,b=geometry.length/2+body.uncertainty,geometry.width/2+body.lateral_padding
    c,s=math.cos(body.yaw),math.sin(body.yaw)
    return [(body.x+x*c-y*s,body.y+x*s+y*c) for x,y in ((a,b),(a,-b),(-a,-b),(-a,b),(a,b))]


def geometry(controller):
    return getattr(controller, '_collision_geometry', BodyGeometry())


def target_body(controller, vehicle_id):
    tracker = controller._v2x_tracker
    samples = tracker._samples.get(vehicle_id)
    if not samples:
        return None
    now = getattr(controller, '_collision_now', samples[-1][0])
    return tracker.collision_body(vehicle_id, now,
        origin=getattr(controller,'_collision_v2x_origin','unconfirmed'),
        offset=getattr(controller,'_collision_center_offset',0.522),
        max_age=getattr(controller,'_collision_max_age',0.5),
        origin_lateral_margin=getattr(controller,'_collision_origin_lateral_margin',None))


def ego_body(controller, pose):
    now = getattr(controller,'_collision_now',0.)
    observation = getattr(controller,'_collision_ego_metadata',None)
    stamp,frame,valid = observation if observation is not None else (now,'map',True)
    alignment = getattr(controller,'_collision_ego_alignment',None)
    if alignment is not None:
        dx,dy,rotation,measured_stamp = alignment
        c,s = math.cos(pose.theta),math.sin(pose.theta)
        return body_pose(float(pose.x)+dx*c-dy*s,float(pose.y)+dx*s+dy*c,
            float(pose.theta)+rotation,measured_stamp,source='measured_center_alignment',direction_valid=True)
    body = body_pose(float(pose.x),float(pose.y),float(pose.theta),stamp,
        frame=frame,source='gnss_position+odom_yaw',direction_valid=True,
        origin=getattr(controller,'_collision_ego_origin','unconfirmed'),
        offset=getattr(controller,'_collision_center_offset',0.522),
        origin_lateral_margin=getattr(controller,'_collision_origin_lateral_margin',None))
    return replace(body,position_valid=body.position_valid and valid)


def predicted_ego(controller, xs, ys, index):
    """Use signed motion to orient rear-axle predictions; stationary keeps actual yaw."""
    a,b=max(index-1,0),min(index+1,len(xs)-1)
    dx,dy=float(xs[b]-xs[a]),float(ys[b]-ys[a])
    yaw=getattr(controller,'_collision_ego_yaw',None)
    if math.hypot(dx,dy)>1e-6:
        direction=math.atan2(dy,dx)
        if yaw is not None and math.cos(direction-yaw)<0:
            direction=math.atan2(math.sin(direction+math.pi),math.cos(direction+math.pi))
        yaw=direction
    from types import SimpleNamespace
    if yaw is None:
        return body_pose(float(xs[index]),float(ys[index]),None,getattr(controller,'_collision_now',0.),origin=getattr(controller,'_collision_ego_origin','unconfirmed'))
    return ego_body(controller,SimpleNamespace(x=xs[index],y=ys[index],theta=yaw))


def swept_path_clear(poses, times, target, velocity, geometry, clearance=0.1):
    """Conservative continuous-segment check, including translation and yaw sweep.

    Times are relative to the target observation. Subdivision bounds cost;
    insufficient/invalid evidence never grants a spacing-control bypass.
    """
    if (len(poses) < 2 or len(times) != len(poses) or not target.position_valid
            or not all(p.position_valid for p in poses)
            or not all(math.isfinite(t) and t >= 0 for t in times)
            or not all(math.isfinite(v) for v in velocity)):
        return False
    vx, vy = velocity
    count = 0
    for a, b, t0, t1 in zip(poses, poses[1:], times, times[1:]):
        if t1 < t0 or not a.yaw_valid or not b.yaw_valid:
            return False
        distance = math.hypot(b.x-a.x, b.y-a.y)
        angle = math.atan2(math.sin(b.yaw-a.yaw), math.cos(b.yaw-a.yaw))
        steps = max(1, math.ceil(distance/.1), math.ceil(abs(angle)/.05), math.ceil((t1-t0)/.1))
        count += steps
        if count > 512:
            return False
        radius = geometry.radius + math.sqrt(2)*max(a.uncertainty, b.uncertainty,
                                                    a.lateral_padding, b.lateral_padding)
        padding = distance/(2*steps) + radius*abs(angle)/(2*steps)
        target_padding = math.hypot(vx,vy)*(t1-t0)/(2*steps)
        for j in range(steps):
            ratio = (j+.5)/steps
            time = t0+(t1-t0)*ratio
            ego = replace(a, x=a.x+(b.x-a.x)*ratio, y=a.y+(b.y-a.y)*ratio,
                          yaw=a.yaw+angle*ratio,
                          uncertainty=max(a.uncertainty,b.uncertainty)+padding,
                          lateral_uncertainty=max(a.lateral_padding,b.lateral_padding)+padding)
            other = replace(target, x=target.x+vx*time, y=target.y+vy*time,
                            uncertainty=target.uncertainty+target_padding,
                            lateral_uncertainty=target.lateral_padding+target_padding)
            if overlaps(ego,other,geometry,margin=clearance) is not False:
                return False
    return True
