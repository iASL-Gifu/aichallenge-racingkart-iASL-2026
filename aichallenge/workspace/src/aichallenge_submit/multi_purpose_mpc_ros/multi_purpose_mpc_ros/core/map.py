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
from copy import deepcopy

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
        self.revision = 0

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

    def static_body_is_free(self, body, geometry, padding=0.0):
        return self._static_body_collision_evidence(body, geometry, padding) is None

    def begin_collision_cycle(self):
        """Cache only immutable static-map footprint evidence for this tick."""
        self._body_collision_cache = {}
        self._recovery_prefix_failures = {}

    def static_body_collision_detail(self, body, geometry, padding=0.0, *, include_cells=False):
        return deepcopy(self._static_body_collision_evidence(body, geometry, padding, include_cells=include_cells))

    def _static_body_collision_evidence(self, body, geometry, padding=0.0, *, include_cells=False):
        """Private borrowed evidence; callers must never mutate it."""
        cache = getattr(self, '_body_collision_cache', None)
        if cache is None:
            return self._compute_static_body_collision_detail(
                body, geometry, padding, include_cells=include_cells)
        key = (id(self.data_backup), self.resolution, tuple(self.origin),
               self.width, self.height, body, geometry, padding, include_cells)
        if key not in cache:
            if len(cache) >= 4096:
                cache.clear()
            cache[key] = self._compute_static_body_collision_detail(
                body, geometry, padding, include_cells=include_cells)
        # Evidence contains mutable sets/lists; callers cannot poison reuse.
        return cache[key]

    def _compute_static_body_collision_detail(self, body, geometry, padding=0.0, *, include_cells=False):
        """Return collision evidence, or None for a free oriented footprint."""
        if not body.position_valid or not body.yaw_valid:
            return {"reason": "invalid_body"}
        hl = geometry.length / 2 + body.uncertainty + padding
        hw = geometry.width / 2 + body.lateral_padding + padding
        c, sn = math.cos(body.yaw), math.sin(body.yaw)
        rx, ry = abs(c)*hl + abs(sn)*hw, abs(sn)*hl + abs(c)*hw
        res = float(self.resolution)
        cx = (body.x-self.origin[0])/res
        cy = self.height-1-(body.y-self.origin[1])/res
        xmin, xmax = math.floor(cx-rx/res-.5), math.ceil(cx+rx/res+.5)
        ymin, ymax = math.floor(cy-ry/res-.5), math.ceil(cy+ry/res+.5)
        detail = dict(center=[body.x, body.y], yaw=body.yaw,
                      checked_length=2*hl, checked_width=2*hw,
                      origin=body.origin, yaw_source=body.yaw_source,
                      uncertainty=body.uncertainty, lateral_uncertainty=body.lateral_padding,
                      pixel_bounds=[xmin, xmax, ymin, ymax])
        if xmin < 0 or ymin < 0 or xmax >= self.width or ymax >= self.height:
            return dict(detail, reason="out_of_map")
        # A free axis-aligned superset proves the oriented footprint is free.
        # Keep out-of-map/invalid-body checks above this fast path.
        occupied_region = self.data_backup[ymin:ymax+1, xmin:xmax+1] == 0
        if not occupied_region.any():
            return None
        iy, ix = np.ogrid[ymin:ymax+1, xmin:xmax+1]
        dx, dy = (ix-cx)*res, -(iy-cy)*res
        cell_padding = .5*res*(abs(c)+abs(sn))
        overlap = ((np.abs(dx*c+dy*sn) <= hl+cell_padding)
                   & (np.abs(-dx*sn+dy*c) <= hw+cell_padding))
        occupied = overlap & occupied_region
        rows, cols = np.nonzero(occupied)
        if len(rows):
            px, py = xmin+int(cols[0]), ymin+int(rows[0])
            if include_cells:
                detail["occupied_cells"] = set(zip((cols+xmin).tolist(), (rows+ymin).tolist()))
                depth = np.minimum(hl+cell_padding-np.abs(dx*c+dy*sn),
                                   hw+cell_padding-np.abs(-dx*sn+dy*c))
                detail["overlap_depth"] = float(depth[occupied].sum())
                detail["max_overlap_depth"] = float(depth[occupied].max())
            return dict(detail, reason="occupied_cell", occupied_count=len(rows),
                        first_pixel=[px, py], first_world=list(self.m2w(px, py)))
        return None

    def static_recovery_path_is_clear(self, bodies, geometry, *, temporary_depth_increase=0., recovery_contact_slide=False, wall_margin=.05):
        """Separate physical contact, wall clearance and inter-sample coverage.

        The same padding is used throughout a path so changing the padding
        itself cannot masquerade as decreasing overlap during recovery.
        """
        from .wall_constraints import swept_sample_padding
        if not math.isfinite(wall_margin) or wall_margin < 0.:
            return False, 'invalid_wall_margin'
        if not bodies:
            return False, 'empty_path'
        sweep = swept_sample_padding(bodies, geometry)
        if not math.isfinite(sweep):
            return False, 'invalid_ego_body'
        safe, reason = self._static_recovery_samples_are_clear(
            bodies, geometry, padding=wall_margin+sweep,
            temporary_depth_increase=temporary_depth_increase,
            recovery_contact_slide=recovery_contact_slide)
        if not safe:
            return False, reason + f', contact=clearance_or_sweep, wall_margin={wall_margin:.3f}, sweep_padding={sweep:.6f}'
        if safe and reason == 'clear':
            # Expanded samples prove both physical and continuous clearance.
            # Avoid a second raster scan for the overwhelmingly common case.
            return True, reason
        # Margin-only overlap cannot authorize a newly contacting real body.
        physical, physical_reason = self._static_recovery_samples_are_clear(
            bodies, geometry, padding=0., temporary_depth_increase=temporary_depth_increase,
            recovery_contact_slide=recovery_contact_slide)
        if not physical:
            return False, physical_reason + ', contact=physical'
        if safe and reason == 'wall_escape' and physical and physical_reason == 'clear':
            for i, (a, b) in enumerate(zip(bodies, bodies[1:]), 1):
                if not self._physical_segment_is_clear(a, b, geometry):
                    return False, f'new_wall_contact_at_step={i}, contact=physical_sweep'
        return safe, reason

    def _physical_segment_is_clear(self, a, b, geometry, depth=0):
        """Prove a margin-only escape never crosses a wall between samples."""
        from dataclasses import replace
        from .wall_constraints import swept_sample_padding
        pad = swept_sample_padding((a, b), geometry)
        if (self.static_body_is_free(a, geometry, pad)
                and self.static_body_is_free(b, geometry, pad)):
            return True
        if depth >= 8:
            return False  # Cannot establish continuous clearance; do not guess.
        angle = math.atan2(math.sin(b.yaw-a.yaw), math.cos(b.yaw-a.yaw))
        mid = replace(a, x=(a.x+b.x)/2, y=(a.y+b.y)/2, yaw=a.yaw+angle/2,
                      uncertainty=max(a.uncertainty,b.uncertainty),
                      lateral_uncertainty=max(a.lateral_padding,b.lateral_padding))
        if not self.static_body_is_free(mid, geometry):
            return False
        return (self._physical_segment_is_clear(a, mid, geometry, depth+1)
                and self._physical_segment_is_clear(mid, b, geometry, depth+1))

    def _static_recovery_samples_are_clear(self, bodies, geometry, *, padding,
                                         temporary_depth_increase=0., recovery_contact_slide=False):
        import re
        cache=getattr(self, '_recovery_prefix_failures', None)
        bodies=tuple(bodies)
        key=(id(self.data_backup), self.resolution, tuple(self.origin), self.width,
             self.height, geometry, padding, temporary_depth_increase, recovery_contact_slide)
        if cache is not None:
            for prefix,result in cache.get(key, ()):
                if len(bodies)>=len(prefix) and bodies[:len(prefix)]==prefix:
                    return result
        result=self._compute_static_recovery_samples_are_clear(
            bodies,geometry,padding=padding,temporary_depth_increase=temporary_depth_increase,
            recovery_contact_slide=recovery_contact_slide)
        # End-of-path improvement is not a prefix property. Only failures at
        # a particular checked sample may veto another identical prefix.
        match=re.search(r'_at_step=(\d+)',result[1]) if not result[0] else None
        if cache is not None and match:
            if len(cache)>=128:cache.clear()
            entries=cache.setdefault(key,[])
            if len(entries)>=64:entries.pop(0)
            entries.append((bodies[:int(match.group(1))+1],result))
        return result

    def _compute_static_recovery_samples_are_clear(self, bodies, geometry, *, padding,
                                         temporary_depth_increase=0., recovery_contact_slide=False):
        if not math.isfinite(temporary_depth_increase) or temporary_depth_increase < 0.:
            return False, 'invalid_overlap_allowance'
        initial_max_depth = initial_depth = previous_depth = 0.
        previous_cells = set()
        previous_max_depth = 0.
        for i, body in enumerate(bodies):
            detail = self._static_body_collision_evidence(body, geometry, padding, include_cells=True)
            if detail is not None and detail['reason'] != 'occupied_cell':
                return False, f"static_collision_at_step={i}, detail={detail}"
            cells = detail['occupied_cells'] if detail else set()
            depth = detail['overlap_depth'] if detail else 0.
            max_depth = detail['max_overlap_depth'] if detail else 0.
            if i == 0:
                initial_depth = depth
                initial_max_depth = max_depth
            else:
                # A 5 cm motion can change the contacted 10 cm map cells even
                # while retreating. New contact must adjoin the previous patch;
                # a separated obstacle or recontact after clearance is rejected.
                added = cells - previous_cells
                disconnected = any(
                    not any((x+dx, y+dy) in previous_cells
                            for dx in (-1,0,1) for dy in (-1,0,1))
                    for x,y in added)
                if disconnected and recovery_contact_slide and initial_depth > 0. and previous_cells:
                    # Trace the occupied wall locally, not through free space.
                    # Three cells bound contact migration; globally connected
                    # walls must not make every new obstacle permissible.
                    reached = set(previous_cells)
                    frontier = reached.copy()
                    for _ in range(3):
                        next_cells = set()
                        for x, y in frontier:
                            for dx in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    nx, ny = x+dx, y+dy
                                    if (0 <= nx < self.width and 0 <= ny < self.height
                                            and self.data_backup[ny, nx] == 0
                                            and (nx, ny) not in reached):
                                        next_cells.add((nx, ny))
                        reached.update(next_cells)
                        frontier = next_cells
                    disconnected = not added.issubset(reached)
                if disconnected:
                    summary = {key: value for key, value in detail.items() if key != 'occupied_cells'}
                    return False, f'new_wall_contact_at_step={i}, padding={padding:.3f}, detail={summary}'
                if temporary_depth_increase > 0. and initial_depth > 0.:
                    if max_depth > initial_max_depth + temporary_depth_increase + 1e-8:
                        return False, (f'wall_overlap_limit_at_step={i}, '
                                       f'initial_max={initial_max_depth:.6f}, '
                                       f'max_depth={max_depth:.6f}, allowance={temporary_depth_increase:.3f}')
                elif depth > previous_depth + 1e-8 or max_depth > previous_max_depth + 1e-8:
                    return False, (f'wall_overlap_increases_at_step={i}, '
                                   f'depth={previous_depth:.6f}->{depth:.6f}, '
                                   f'max_depth={previous_max_depth:.6f}->{max_depth:.6f}')
            previous_cells, previous_depth = cells, depth
            previous_max_depth = max_depth
        if initial_depth > 0.:
            if (previous_depth >= initial_depth - 1e-6
                    or previous_max_depth > initial_max_depth + 1e-8):
                return False, 'wall_overlap_not_reduced'
            return True, 'wall_escape'
        return True, 'clear'

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
        self.revision += 1
        self.data = self.data_backup.copy()
        self.obstacles = list()

    def add_obstacles(self, obstacles):
        """
        Add obstacles to the map.
        :param obstacles: list of obstacle objects
        """

        # Extend list of obstacles
        self.revision += 1
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
        self.revision += 1
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
