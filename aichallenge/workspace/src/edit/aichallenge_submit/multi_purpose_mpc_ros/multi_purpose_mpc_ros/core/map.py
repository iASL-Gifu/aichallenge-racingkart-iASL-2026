import numpy as np
from os import path
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage.morphology import remove_small_holes
from skimage.draw import line_aa
import matplotlib.patches as plt_patches
import math

# Colors
OBSTACLE = '#2E4053'


############
# Obstacle #
############

class Obstacle:
    def __init__(self, cx, cy, radius):
        """
        Constructor for a circular obstacle to be placed on a map.
        :param cx: x coordinate of center of obstacle in world coordinates
        :param cy: y coordinate of center of obstacle in world coordinates
        :param radius: radius of circular obstacle in m
        """
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def show(self, ax):
        """
        Display obstacle on the provided axis.
        :param ax: Matplotlib axis object to plot on
        """
        # Draw circle
        circle = plt_patches.Circle(xy=(self.cx, self.cy), radius=self.radius,
                                    color=OBSTACLE, zorder=20)
        ax.add_patch(circle)


#######
# Map #
#######

class Map:
    def __init__(self, map_yaml_path):
        """
        Constructor for map object. Map contains occupancy grid map data of
        environment as well as meta information.
        :param map_yaml_path: path to map yaml
        """

        base_path = path.dirname(map_yaml_path)
        base_name = path.splitext(path.basename(map_yaml_path))[0]
        with open(map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        # Set binarization threshold
        self.threshold_occupied = map_data['occupied_thresh']

        pgm_file_path = path.join(base_path, map_data['image'])
        image = mpimg.imread(pgm_file_path)
        image_array = np.array(image)

        # file_path = path.join(base_path, map_data['image']).replace('pgm', 'png')
        # image = Image.open(file_path)
        # image_array = np.array(image)

        # Numpy array containing map data
        if image_array.ndim == 3:
            self.data = image_array[:, :, 0]
        elif image_array.ndim == 2:
            self.data = image_array
        else:
            raise ValueError("Unexpected image dimensions")

        # Process raw map image
        self.process_map()

        # Store meta information
        self.height = self.data.shape[0]  # height of the map in px
        self.width = self.data.shape[1]  # width of the map in px
        self.resolution = map_data['resolution']  # resolution of the map in m/px
        self.origin = map_data['origin']  # x and y coordinates of map origin
        # (bottom-left corner) in m

        # Containers for user-specified additional obstacles and boundaries
        self.obstacles = list()
        self.boundaries = list()

        self.data_backup = self.data.copy()

    def w2m(self, x, y):
        """
        World2Map. Transform coordinates from global coordinate system to
        map coordinates.
        :param x: x coordinate in global coordinate system
        :param y: y coordinate in global coordinate system
        :return: discrete x and y coordinates in px
        """
        # dx = int(np.floor((x - self.origin[0]) / self.resolution))
        # dy = int(np.floor((y - self.origin[1]) / self.resolution))

        dx = int((x - self.origin[0]) / self.resolution + 0.5)
        dy = int((self.height - 1) - (y - self.origin[1]) / self.resolution + 0.5)
        dx = np.clip(dx, 0, self.width - 1)
        dy = np.clip(dy, 0, self.height - 1)

        return dx, dy

    def m2w(self, dx, dy):
        """
        Map2World. Transform coordinates from map coordinate system to
        global coordinates.
        :param dx: x coordinate in map coordinate system
        :param dy: y coordinate in map coordinate system
        :return: x and y coordinates of cell center in global coordinate system
        """
        x = int(dx + 0.5) * self.resolution + self.origin[0]
        y = (self.height - 1 - int(dy + 0.5)) * self.resolution + self.origin[1]

        return x, y

    def static_disk_is_free(self, x, y, radius):
        """Return whether a world-space disk is inside static free space.

        Out-of-map samples are occupied. ``data_backup`` excludes dynamic V2X
        obstacles, which are checked separately by the rear-traffic logic.
        """
        radius = max(float(radius), 0.0)
        resolution = float(self.resolution)
        center_x = (float(x) - self.origin[0]) / resolution
        center_y_from_bottom = (float(y) - self.origin[1]) / resolution
        center_y = (self.height - 1) - center_y_from_bottom
        radius_px = int(math.ceil(radius / resolution))
        min_x = int(math.floor(center_x - radius_px))
        max_x = int(math.ceil(center_x + radius_px))
        min_y = int(math.floor(center_y - radius_px))
        max_y = int(math.ceil(center_y + radius_px))
        if (
            min_x < 0 or min_y < 0
            or max_x >= self.width or max_y >= self.height
        ):
            return False

        radius_sq = (radius / resolution) ** 2
        map_y, map_x = np.ogrid[min_y:max_y + 1, min_x:max_x + 1]
        disk = (
            (map_x - center_x) ** 2 + (map_y - center_y) ** 2
            <= radius_sq
        )
        cells = self.data_backup[min_y:max_y + 1, min_x:max_x + 1]
        return bool(np.all(cells[disk] != 0))

    def static_oriented_box_is_free(
        self, x, y, heading, half_length, half_width, step=None
    ):
        """Return whether an oriented rectangular footprint is in free space.

        This is intentionally separate from ``static_disk_is_free``. A disk
        enclosing the complete kart is useful for conservative reverse sweeps,
        but rejects valid wall-escape motion when the kart is nearly parallel
        to a boundary. Recovery rollouts need the actual footprint orientation.
        """
        half_length = max(float(half_length), 0.0)
        half_width = max(float(half_width), 0.0)
        if step is None:
            step = min(max(float(self.resolution) * 0.75, 0.02), 0.08)
        step = max(float(step), 0.01)
        n_long = max(int(math.ceil(2.0 * half_length / step)), 1)
        n_lat = max(int(math.ceil(2.0 * half_width / step)), 1)
        local_long = np.linspace(-half_length, half_length, n_long + 1)
        local_lat = np.linspace(-half_width, half_width, n_lat + 1)
        longitudinal, lateral = np.meshgrid(local_long, local_lat)
        cos_yaw = math.cos(float(heading))
        sin_yaw = math.sin(float(heading))
        world_x = (
            float(x) + longitudinal * cos_yaw - lateral * sin_yaw)
        world_y = (
            float(y) + longitudinal * sin_yaw + lateral * cos_yaw)
        map_x = np.rint(
            (world_x - self.origin[0]) / float(self.resolution)).astype(int)
        map_y = np.rint(
            (self.height - 1)
            - (world_y - self.origin[1]) / float(self.resolution)).astype(int)
        if (
            np.any(map_x < 0) or np.any(map_x >= self.width)
            or np.any(map_y < 0) or np.any(map_y >= self.height)
        ):
            return False
        return bool(np.all(self.data_backup[map_y, map_x] != 0))

    def static_oriented_box_interference(
        self, x, y, heading, half_length, half_width, step=None
    ):
        """Return the occupied fraction of an oriented vehicle footprint.

        A value of zero means that the complete footprint is in free space.
        Samples outside the occupancy map are treated as occupied.  The
        fraction, rather than only a boolean, lets recovery distinguish motion
        that exits an existing boundary contact from motion that makes it
        worse.
        """
        half_length = max(float(half_length), 0.0)
        half_width = max(float(half_width), 0.0)
        if step is None:
            step = min(max(float(self.resolution) * 0.75, 0.02), 0.08)
        step = max(float(step), 0.01)
        n_long = max(int(math.ceil(2.0 * half_length / step)), 1)
        n_lat = max(int(math.ceil(2.0 * half_width / step)), 1)
        local_long = np.linspace(-half_length, half_length, n_long + 1)
        local_lat = np.linspace(-half_width, half_width, n_lat + 1)
        longitudinal, lateral = np.meshgrid(local_long, local_lat)
        cos_yaw = math.cos(float(heading))
        sin_yaw = math.sin(float(heading))
        world_x = float(x) + longitudinal * cos_yaw - lateral * sin_yaw
        world_y = float(y) + longitudinal * sin_yaw + lateral * cos_yaw
        map_x = np.rint(
            (world_x - self.origin[0]) / float(self.resolution)).astype(int)
        map_y = np.rint(
            (self.height - 1)
            - (world_y - self.origin[1]) / float(self.resolution)).astype(int)
        inside = (
            (map_x >= 0) & (map_x < self.width)
            & (map_y >= 0) & (map_y < self.height)
        )
        occupied = np.ones(map_x.shape, dtype=bool)
        occupied[inside] = self.data_backup[
            map_y[inside], map_x[inside]] == 0
        return float(np.count_nonzero(occupied)) / float(occupied.size)

    def static_straight_reverse_box_clearance(
        self, x, y, heading, max_distance, half_length, half_width,
        escape_max_distance, step=None, interference_tolerance=0.005,
        trend_window=4, total_improvement_threshold=0.001,
    ):
        """Measure safe straight reverse travel with boundary-contact escape.

        If the initial rectangle is already touching the static boundary, the
        overlap must never increase and must become zero within
        ``escape_max_distance``.  After reaching free space, every remaining
        sample must stay completely free.  Steering is deliberately not
        modelled here.
        """
        max_distance = max(float(max_distance), 0.0)
        escape_max_distance = max(float(escape_max_distance), 0.0)
        interference_tolerance = max(float(interference_tolerance), 0.0)
        trend_window = max(int(trend_window), 2)
        total_improvement_threshold = max(
            float(total_improvement_threshold), 0.0)
        if step is None:
            step = min(max(float(self.resolution) * 0.5, 0.02), 0.10)
        step = max(float(step), 0.01)
        initial = self.static_oriented_box_interference(
            x, y, heading, half_length, half_width)
        diagnostic = {
            "reason": "free_path",
            "initial_interference": initial,
            "failure_distance": None,
            "previous_interference": initial,
            "interference": initial,
            "free_distance": 0.0 if initial <= 0.0 else None,
            "safe_prefix_clearance": 0.0,
            "total_improvement": 0.0,
        }
        escaping = initial > 0.0
        reached_free = not escaping
        previous = initial
        recent_interference = [initial]
        clearance = 0.0
        distance = step
        while distance <= max_distance + 1e-9:
            sample_distance = min(distance, max_distance)
            sample_x = float(x) - sample_distance * math.cos(float(heading))
            sample_y = float(y) - sample_distance * math.sin(float(heading))
            interference = self.static_oriented_box_interference(
                sample_x, sample_y, heading, half_length, half_width)
            if not reached_free:
                recent_interference.append(interference)
                recent_interference = recent_interference[-trend_window:]
                total_improvement = initial - interference
                exceeds_initial = (
                    interference > initial + interference_tolerance)
                sustained_worsening = False
                if len(recent_interference) >= trend_window:
                    indices = np.arange(len(recent_interference), dtype=float)
                    slope = float(np.polyfit(
                        indices, np.asarray(recent_interference), 1)[0])
                    window_rise = (
                        recent_interference[-1] - min(recent_interference))
                    sustained_worsening = bool(
                        slope > 0.0
                        and window_rise > interference_tolerance
                        and total_improvement
                            < total_improvement_threshold)
                if exceeds_initial or sustained_worsening:
                    diagnostic.update(
                        reason=(
                            "interference_exceeds_initial"
                            if exceeds_initial
                            else "sustained_interference_worsening"),
                        failure_distance=sample_distance,
                        previous_interference=previous,
                        interference=interference,
                        total_improvement=total_improvement,
                        interference_tolerance=interference_tolerance,
                        trend_window=trend_window,
                    )
                    break
                if interference <= 0.0:
                    reached_free = True
                    diagnostic["free_distance"] = sample_distance
                elif sample_distance > escape_max_distance + 1e-9:
                    diagnostic.update(
                        reason="escape_distance_exceeded",
                        failure_distance=sample_distance,
                        previous_interference=previous,
                        interference=interference,
                        total_improvement=total_improvement,
                        safe_prefix_clearance=sample_distance,
                    )
                    break
            elif interference > 0.0:
                diagnostic.update(
                    reason="collision_after_escape",
                    failure_distance=sample_distance,
                    previous_interference=previous,
                    interference=interference,
                )
                break
            previous = interference
            clearance = sample_distance
            diagnostic["safe_prefix_clearance"] = clearance
            if sample_distance >= max_distance:
                break
            distance += step

        if escaping and not reached_free:
            clearance = 0.0
        elif reached_free and diagnostic["reason"] == "free_path":
            diagnostic["reason"] = "clear"
        self.last_static_reverse_diagnostic = diagnostic
        return clearance, initial, reached_free

    def static_straight_path_clearance(
        self, x, y, heading, max_distance, footprint_radius, step=None
    ):
        """Measure collision-free straight reverse travel on the static map."""
        max_distance = max(float(max_distance), 0.0)
        if step is None:
            step = min(max(float(self.resolution) * 0.5, 0.02), 0.10)
        step = max(float(step), 0.01)
        if not self.static_disk_is_free(x, y, footprint_radius):
            return 0.0

        clearance = 0.0
        distance = step
        while distance <= max_distance + 1e-9:
            sample_distance = min(distance, max_distance)
            sample_x = float(x) - sample_distance * math.cos(float(heading))
            sample_y = float(y) - sample_distance * math.sin(float(heading))
            if not self.static_disk_is_free(
                sample_x, sample_y, footprint_radius
            ):
                break
            clearance = sample_distance
            if sample_distance >= max_distance:
                break
            distance += step
        return clearance

    def process_map(self):
        """
        Process raw map image. Binarization and removal of small holes in map.
        """

        # Binarization using specified threshold
        # 1 corresponds to free, 0 to occupied
        self.data = np.where(self.data >= self.threshold_occupied, 1, 0)

        # Remove small holes in map corresponding to spurious measurements
        self.data = remove_small_holes(self.data, area_threshold=5,
                                       connectivity=8).astype(np.int8)

    def reset_map(self):
        self.data = self.data_backup.copy()
        self.obstacles = list()

    def add_obstacles(self, obstacles):
        """
        Add obstacles to the map.
        :param obstacles: list of obstacle objects
        """

        # Extend list of obstacles
        self.obstacles.extend(obstacles)

        # Iterate over list of new obstacles
        for obstacle in obstacles:

            # Compute radius of circular object in pixels
            radius_px = int(np.ceil(obstacle.radius / self.resolution))
            # Get center coordinates of obstacle in map coordinates
            cx_px, cy_px = self.w2m(obstacle.cx, obstacle.cy)

            # Add circular object to map
            y, x = np.ogrid[-radius_px: radius_px, -radius_px: radius_px]
            index = x ** 2 + y ** 2 <= radius_px ** 2
            self.data[cy_px-radius_px:cy_px+radius_px, cx_px-radius_px:
                                                cx_px+radius_px][index] = 0

    def add_boundary(self, boundaries):
        """
        Add boundaries to the map.
        :param boundaries: list of tuples containing coordinates of boundaries'
        start and end points
        """

        # Extend list of boundaries
        self.boundaries.extend(boundaries)

        # Iterate over list of boundaries
        for boundary in boundaries:
            sx = self.w2m(boundary[0][0], boundary[0][1])
            gx = self.w2m(boundary[1][0], boundary[1][1])
            path_x, path_y, _ = line_aa(sx[0], sx[1], gx[0], gx[1])
            for x, y in zip(path_x, path_y):
                self.data[y, x] = 0


if __name__ == '__main__':
    map = Map('maps/real_map.png')
    # map = Map('maps/sim_map.png')
    plt.imshow(np.flipud(map.data), cmap='gray')
    plt.show()
