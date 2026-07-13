import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 読み込み
# ===========================
data = np.loadtxt(
    "waypoint_bounds_center.csv",
    delimiter=",",
    skiprows=1
)

# 列構造
# 0 idx
# 1 ub
# 2 lb
# 3 left_idx
# 4 right_idx
# 5 left_x
# 6 left_y
# 7 right_x
# 8 right_y
# 9 center_x
#10 center_y

idx = data[:, 0]

left_x  = data[:, 5]
left_y  = data[:, 6]

right_x = data[:, 7]
right_y = data[:, 8]

cx = data[:, 9]
cy = data[:, 10]

# centerは別で必要（もしCSVに無いなら元trajectoryから読む）
# 例：traj_race_cl_mpc.csv
traj = np.loadtxt("min_curv/traj_center.csv", delimiter=",", skiprows=1)
cx = traj[:, 1]
cy = traj[:, 2]

# ===========================
# 閉ループ化
# ===========================
left_x  = np.append(left_x, left_x[0])
left_y  = np.append(left_y, left_y[0])

right_x = np.append(right_x, right_x[0])
right_y = np.append(right_y, right_y[0])

cx = np.append(cx, cx[0])
cy = np.append(cy, cy[0])

# ===========================
# 描画
# ===========================
step = 3

for i in range(0, len(cx)-1, step):  # 最後は閉ループ用なので除く
    plt.scatter(cx[i], cy[i], c='g', s=20)
    plt.text(
        cx[i],
        cy[i],
        str(i),
        fontsize=8,
        color='green'
    )
#plt.figure(figsize=(10, 10))

# center
plt.plot(cx, cy, 'k', lw=2, label="center")

# 左右境界（ここが目的）
plt.plot(left_x, left_y, 'r', lw=2, label="left boundary")
plt.plot(right_x, right_y, 'b', lw=2, label="right boundary")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Reconstructed Track from CSV")

plt.show()