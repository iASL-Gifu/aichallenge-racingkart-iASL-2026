import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

with open("final_ver4/occupancy_grid_map.yaml") as f:
    info = yaml.safe_load(f)

origin_x = info["origin"][0]
origin_y = info["origin"][1]
res = info["resolution"]

img = np.array(Image.open("final_ver3/occupancy_grid_map.pgm"))

traj = np.loadtxt(
    "env/min_curv/traj_race_cl_mpc.csv",
    delimiter=",",
    skiprows=1
)

x = traj[:, 1]
y = traj[:, 2]

# map座標 → 画像座標
px = (x - origin_x) / res
py = img.shape[0] - (y - origin_y) / res

# waypoint index
idx = np.arange(len(traj))

# 大きな画像
plt.figure(figsize=(16, 16))

plt.imshow(img, cmap="gray", origin="upper")
plt.plot(px, py, "r-", linewidth=2)

# 5 waypointごとに番号表示
for i in range(0, len(idx), 5):
    plt.text(
        px[i],
        py[i],
        str(idx[i]),
        fontsize=8,
        color="blue"
    )

# 開始点と終了点
plt.scatter(px[0], py[0], s=80, marker="o")
plt.scatter(px[-1], py[-1], s=80, marker="x")

plt.title("traj_mincurv waypoint index")
plt.axis("equal")
plt.tight_layout()
plt.show()
