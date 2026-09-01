"""Regression tests for cached RViz lane-bound marker geometry."""

from types import SimpleNamespace

import multi_purpose_mpc_ros.mpc_controller as controller_module
from multi_purpose_mpc_ros.mpc_controller import MPCController


class _Message:
    def __init__(self, **values):
        self.header = SimpleNamespace(frame_id=None)
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(w=0.0),
        )
        self.scale = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        for name, value in values.items():
            setattr(self, name, value)


class _Marker(_Message):
    TRIANGLE_LIST = 11
    ADD = 0

    def __init__(self, **values):
        super().__init__(**values)
        self.points = []
        self.colors = []


class _MarkerArray:
    def __init__(self):
        self.markers = []


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class _ReferencePath:
    def __init__(self):
        self.n_waypoints = 3
        self.circular = True
        self.target_lane_idx = None
        self.overtake_zone = [True, False, True]
        self.lane_bounds_calls = 0
        self.waypoints = [
            SimpleNamespace(
                x=float(index), y=0.0, psi=0.0, normal_angle=0.0,
                ub=3.0, lb=-3.0,
            )
            for index in range(self.n_waypoints)
        ]

    def get_waypoint(self, index):
        return self.waypoints[index % self.n_waypoints]

    def get_lane_bounds(self, _index, _n_lanes=3):
        self.lane_bounds_calls += 1
        return [(-0.5, -3.0), (0.5, -0.5), (3.0, 0.5)]


def _marker_snapshot(marker_array):
    return [
        {
            "id": marker.id,
            "ns": marker.ns,
            "points": [
                (point.x, point.y, point.z) for point in marker.points],
            "colors": [
                (color.r, color.g, color.b, color.a)
                for color in marker.colors
            ],
        }
        for marker in marker_array.markers
    ]


def test_lane_marker_cache_preserves_marker_content(monkeypatch):
    monkeypatch.setattr(controller_module, "Marker", _Marker)
    monkeypatch.setattr(controller_module, "MarkerArray", _MarkerArray)
    monkeypatch.setattr(controller_module, "Point", _Message)
    monkeypatch.setattr(controller_module, "Vector3", _Message)
    monkeypatch.setattr(controller_module, "ColorRGBA", _Message)

    controller = MPCController.__new__(MPCController)
    controller._map_z = 0.0
    controller._lane_marker_pub = _Publisher()
    path = _ReferencePath()

    controller._publish_lane_markers(path, use_cache=False)
    uncached = controller._lane_marker_pub.messages[-1]
    path.lane_bounds_calls = 0

    controller._publish_lane_markers(path)
    first_cached = controller._lane_marker_pub.messages[-1]
    first_build_calls = path.lane_bounds_calls
    controller._publish_lane_markers(path)
    second_cached = controller._lane_marker_pub.messages[-1]

    assert _marker_snapshot(first_cached) == _marker_snapshot(uncached)
    assert second_cached is first_cached
    assert len(controller._lane_marker_pub.messages) == 2
    assert path.lane_bounds_calls == first_build_calls

    path.target_lane_idx = 2
    controller._publish_lane_markers(path)
    assert controller._lane_marker_pub.messages[-1] is not first_cached
    assert path.lane_bounds_calls > first_build_calls

    # Returning to an older cached key must republish it because the transient
    # publisher's latest sample currently contains the L2-highlighted markers.
    target_two_build_calls = path.lane_bounds_calls
    path.target_lane_idx = None
    controller._publish_lane_markers(path)
    assert controller._lane_marker_pub.messages[-1] is first_cached
    assert len(controller._lane_marker_pub.messages) == 4
    assert path.lane_bounds_calls == target_two_build_calls


def test_map_boundary_update_invalidates_lane_marker_cache():
    class _BoundaryPath:
        def update_boundaries_from_markers(self, _left, _right):
            pass

    controller = MPCController.__new__(MPCController)
    controller._lane_marker_cache = {("old",): object()}
    controller._lane_marker_last_published_cache_key = ("old",)
    controller._reference_pathN_race = _BoundaryPath()
    controller._reference_pathN_center = _BoundaryPath()
    controller._map_marker_sub = object()
    controller.destroy_subscription = lambda _subscription: None
    controller.get_logger = lambda: SimpleNamespace(info=lambda *_args: None)
    message = SimpleNamespace(markers=[
        SimpleNamespace(ns="left_lane_bound", points=[_Message(x=0.0, y=1.0)]),
        SimpleNamespace(ns="right_lane_bound", points=[_Message(x=0.0, y=-1.0)]),
    ])

    controller._map_marker_callback(message)

    assert controller._lane_marker_cache == {}
    assert controller._lane_marker_last_published_cache_key is None


def test_lane_marker_timing_separates_cache_miss_and_hit(monkeypatch):
    monkeypatch.setattr(controller_module, "Marker", _Marker)
    monkeypatch.setattr(controller_module, "MarkerArray", _MarkerArray)
    monkeypatch.setattr(controller_module, "Point", _Message)
    monkeypatch.setattr(controller_module, "Vector3", _Message)
    monkeypatch.setattr(controller_module, "ColorRGBA", _Message)

    controller = MPCController.__new__(MPCController)
    controller._map_z = 0.0
    controller._loop = 40
    controller._mpc_cfg = SimpleNamespace(control_rate=40)
    controller._lane_marker_pub = _Publisher()
    logger = _Logger()
    controller.get_logger = lambda: logger
    path = _ReferencePath()

    controller._publish_lane_markers(path)
    controller._publish_lane_markers(path)

    assert len(logger.messages) == 2
    assert "[LaneMarkerTiming]" in logger.messages[0]
    assert "loop=40" in logger.messages[0]
    assert "cache_hit=False" in logger.messages[0]
    assert "cache_miss=True" in logger.messages[0]
    assert "geometry_build_ms=" in logger.messages[0]
    assert "publish_ms=" in logger.messages[0]
    assert "marker_count=3" in logger.messages[0]
    assert "publish_skipped=False" in logger.messages[0]
    assert "cache_hit=True" in logger.messages[1]
    assert "cache_miss=False" in logger.messages[1]
    assert "geometry_build_ms=0.000" in logger.messages[1]
    assert "publish_skipped=True" in logger.messages[1]
    assert len(controller._lane_marker_pub.messages) == 1
