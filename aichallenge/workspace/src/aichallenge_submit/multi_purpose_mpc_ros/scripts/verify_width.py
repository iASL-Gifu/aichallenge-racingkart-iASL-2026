#!/usr/bin/env python3

import os
import sys

# Add multi_purpose_mpc_ros to python path
sys.path.append("/home/haruki/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros")

from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.utils import load_ref_path

def main():
    base_dir = "/home/haruki/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver4"
    yaml_path = os.path.join(base_dir, "occupancy_grid_map.yaml")
    csv_path = os.path.join(base_dir, "traj_mincurv.csv")

    print("Loading map...")
    m = Map(yaml_path)
    print("Loading waypoints...")
    wp_x, wp_y, _, _ = load_ref_path(csv_path)

    print("Building ReferencePath...")
    ref_path = ReferencePath(
        m, wp_x, wp_y,
        resolution=0.6,
        smoothing_distance=2,
        max_width=6.0,
        circular=True
    )

    print(f"\nTotal constructed waypoints: {ref_path.n_waypoints}")
    print("\n--- Waypoint Boundaries (wp 30 to 50) ---")
    print(f"{'WP ID':<6} | {'X':<10} | {'Y':<10} | {'Left Bound (ub)':<16} | {'Right Bound (lb)':<17} | {'Total Width (m)':<15}")
    print("-" * 88)

    for idx in range(30, 51):
        wp = ref_path.get_waypoint(idx)
        width = wp.ub - wp.lb
        print(f"{idx:<6} | {wp.x:<10.3f} | {wp.y:<10.3f} | {wp.ub:<16.3f} | {wp.lb:<17.3f} | {width:<15.3f}")

if __name__ == "__main__":
    main()
