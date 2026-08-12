import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# CSV読み込み
# ==========================
csv_file = "centerline_devided_recomputed.csv"   # ←変更
print("centerline_devided_recomputed.csv")

df = pd.read_csv(csv_file)

cx = df["center_x"].to_numpy()
cy = df["center_y"].to_numpy()

left_width = df["w_tr_left_m"].to_numpy()
right_width = df["w_tr_right_m"].to_numpy()

# ==========================
# 接線ベクトル
# ==========================
dx = np.gradient(cx)
dy = np.gradient(cy)

length = np.hypot(dx, dy)

tx = dx / length
ty = dy / length

# ==========================
# 左向き法線ベクトル
# ==========================
nx = -ty
ny = tx

# ==========================
# 左右境界生成
# ==========================
left_x = cx + nx * left_width
left_y = cy + ny * left_width

right_x = cx - nx * right_width
right_y = cy - ny * right_width

# ==========================
# 描画
# ==========================
plt.figure(figsize=(10,10))

plt.plot(cx, cy, 'k--', linewidth=1.5, label="Center")

plt.plot(left_x, left_y, 'b', linewidth=2, label="Left Boundary")
plt.plot(right_x, right_y, 'r', linewidth=2, label="Right Boundary")

plt.fill(
    np.r_[left_x, right_x[::-1]],
    np.r_[left_y, right_y[::-1]],
    color="lightgray",
    alpha=0.5
)

plt.axis("equal")
plt.grid(True)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.title("Track reconstructed from centerline and widths")

plt.show()