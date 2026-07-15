#!/usr/bin/env python3

import os
import yaml
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

def main():
    # File paths
    base_dir = "/home/haruki/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver4"
    yaml_path = os.path.join(base_dir, "occupancy_grid_map.yaml")
    pgm_path = os.path.join(base_dir, "occupancy_grid_map.pgm")
    csv_path = os.path.join(base_dir, "traj_mincurv.csv")
    output_path = os.path.join(base_dir, "map_trajectory_overlay.png")

    print(f"Loading YAML: {yaml_path}")
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    resolution = meta['resolution']
    origin_x = meta['origin'][0]
    origin_y = meta['origin'][1]
    print(f"Resolution: {resolution} m/pixel")
    print(f"Origin: ({origin_x}, {origin_y})")

    print(f"Loading Image: {pgm_path}")
    img = Image.open(pgm_path).convert("RGB")
    width, height = img.size
    print(f"Image dimensions: {width}x{height}")

    print(f"Loading Trajectory: {csv_path}")
    df = pd.read_csv(csv_path)
    wps_x = df['x_m'].values
    wps_y = df['y_m'].values

    # Convert coordinates to pixel space
    # In ROS occupancy grid, origin is the bottom-left corner of the map.
    # In image coordinate space, origin (0, 0) is the top-left corner.
    pixels_x = ((wps_x - origin_x) / resolution).astype(int)
    pixels_y = (height - 1 - (wps_y - origin_y) / resolution).astype(int)

    # Check boundaries and draw
    draw = ImageDraw.Draw(img)
    
    occupied_count = 0
    unknown_count = 0
    free_count = 0

    print("Checking waypoint occupancy status...")
    for idx, (px, py) in enumerate(zip(pixels_x, pixels_y)):
        if 0 <= px < width and 0 <= py < height:
            # Check pixel color in original grayscale image (loaded directly)
            orig_img = Image.open(pgm_path)
            pixel_val = orig_img.getpixel((px, py))
            
            # Map representation: 254/255 is free (white), 0 is occupied (black), 205 is unknown (gray)
            if pixel_val == 0:
                occupied_count += 1
                status = "OCCUPIED (Wall/Obstacle) 🚨"
            elif pixel_val == 205:
                unknown_count += 1
                status = "UNKNOWN"
            else:
                free_count += 1
                status = "FREE"

            # Draw a circle on the overlay image
            # Red color for standard waypoints
            color = (255, 0, 0)
            if idx == 0:
                color = (0, 255, 0)  # Green for start/index 0
            
            # Draw a dot of radius 2
            draw.ellipse([px-2, py-2, px+2, py+2], fill=color)
        else:
            print(f"Warning: Waypoint {idx} is outside the map boundaries (pixel: {px}, {py})")

    # Save overlay image
    img.save(output_path)
    print(f"\nOverlay image successfully saved to: {output_path}")
    print("\nSummary:")
    print(f"Total Waypoints: {len(df)}")
    print(f"  - In FREE space (drivable): {free_count}")
    print(f"  - In OCCUPIED space (wall): {occupied_count}")
    print(f"  - In UNKNOWN space: {unknown_count}")

    if occupied_count > 0:
        print("\n🚨 WARNING: Some waypoints are inside OCCUPIED (black) areas! The racing line crosses walls.")
    else:
        print("\n🟢 SUCCESS: All waypoints are in FREE (white) drivable space. The trajectory is safe.")

if __name__ == "__main__":
    main()
