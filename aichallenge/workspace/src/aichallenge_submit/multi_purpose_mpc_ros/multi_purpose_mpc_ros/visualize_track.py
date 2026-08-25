import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
_dir = "/home/takenoyama/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env"
TRACK_CSV = os.path.join(_dir, "track.csv")
OUT_IMG = "//home/takenoyama/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/plot_track.png"

# Load data
print(f"Loading track.csv from {TRACK_CSV}")
track = np.loadtxt(TRACK_CSV, delimiter=",", skiprows=1)

t_x  = track[:, 0]
t_y  = track[:, 1]
t_wr = track[:, 2]   # right half-width
t_wl = track[:, 3]   # left half-width

# Calculate orientation and normals
dx = np.roll(t_x, -1) - np.roll(t_x, 1)
dy = np.roll(t_y, -1) - np.roll(t_y, 1)
norm_d = np.sqrt(dx**2 + dy**2) + 1e-8
lnx = -dy / norm_d
lny =  dx / norm_d

# Calculate boundary points
left_wall_x = t_x + lnx * t_wl
left_wall_y = t_y + lny * t_wl
right_wall_x = t_x - lnx * t_wr
right_wall_y = t_y - lny * t_wr

# Create plot
plt.figure(figsize=(10, 10))
plt.plot(t_x, t_y, 'g--', label='Centerline')
plt.plot(left_wall_x, left_wall_y, 'r-', label='Left Boundary')
plt.plot(right_wall_x, right_wall_y, 'b-', label='Right Boundary')
plt.scatter(t_x[0], t_y[0], color='black', marker='o', s=100, label='Start Point')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Track Visualization (Centerline & Boundaries)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Save plot
os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
plt.savefig(OUT_IMG, dpi=300)
print(f"Saved plot to: {OUT_IMG}")

