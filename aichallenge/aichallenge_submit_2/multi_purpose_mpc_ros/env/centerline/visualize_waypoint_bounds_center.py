import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#========================
# CSV読み込み
#========================
traj = pd.read_csv("traj_center313.csv")
bound = pd.read_csv("waypoint_bounds_center.csv")

#========================
# データ取得
#========================
x = traj["x_m"].values
y = traj["y_m"].values
psi = traj["psi_rad"].values

ub = bound["ub"].values
lb = bound["lb"].values

#========================
# 法線ベクトル
#========================
nx = -np.cos(psi)
ny = -np.sin(psi)

#========================
# 左境界
#========================
left_x = x + nx * ub
left_y = y + ny * ub

#========================
# 右境界
#========================
right_x = x + nx * lb
right_y = y + ny * lb

#========================
# 描画
#========================]
print(nx[100], ny[100])



# センターライン
plt.plot(x, y, 'k', linewidth=2, label="Center Line")

# 左右境界
plt.plot(left_x, left_y, 'r', linewidth=2, label="Left Boundary")
plt.plot(right_x, right_y, 'b', linewidth=2, label="Right Boundary")

# 等倍率
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.show()