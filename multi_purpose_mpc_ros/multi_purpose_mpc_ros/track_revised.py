"""
track_revised.py
================
env/track.csv を読み込み、create_track.py の内部可視化で使われている
ガウシアン平滑化法線を用いて綺麗な境界座標を復元し、
それを track_revised.csv として保存する。

出力フォーマット: x_m, y_m, w_tr_right_m, w_tr_left_m
  (visualize_track.py や generate_waypoint_bounds.py がそのまま読み込める形式)

使い方:
  python3 track_revised.py

出力:
  ../env/track_revised.csv  (綺麗な境界形状を反映した track.csv 互換ファイル)
"""

#from multi_purpose_mpc_ros.multi_purpose_mpc_ros.create_track import width
import numpy as np
import csv
import math
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ─── パス設定 ──────────────────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
IN_CSV  = os.path.join(_dir, "../env/track.csv")
OUT_CSV = os.path.join(_dir, "../env/track_revised.csv")
OUT_PNG = os.path.join(_dir, "../env/plot_track_revised.png")

# ─── track.csv を読み込む ──────────────────────────────────────────────────
print(f"Loading: {IN_CSV}")
track = np.loadtxt(IN_CSV, delimiter=",", skiprows=1)

cx  = track[:, 0]   # センターライン X
cy  = track[:, 1]   # センターライン Y
w_r = track[:, 2]   # 右側幅 (w_tr_right_m)
w_l = track[:, 3]   # 左側幅 (w_tr_left_m)


N = len(cx)
print(f"  {N} points loaded")

# ─── ガウシアン平滑化法線を計算 (create_track.py と同じ) ──────────────────
# create_track.py の nx0/ny0 は右向き法線 (dy/norm, -dx/norm)
dx0 = gaussian_filter1d(cx, sigma=2, order=1, mode='wrap')
dy0 = gaussian_filter1d(cy, sigma=2, order=1, mode='wrap')
norm0 = np.sqrt(dx0**2 + dy0**2) + 1e-8
nx0 =  dy0 / norm0   # 右向き (create_track.py の元の定義)
ny0 = -dx0 / norm0

# ─── create_track.py の可視化ブロックと同じ計算で境界点を復元 ──────────────
# csv_left  = center + n * w_left   (w_left は負になり得る → 左方向)
# csv_right = center - n * w_right  (右方向)
csv_left_x  = cx + nx0 * w_l
csv_left_y  = cy + ny0 * w_l
csv_right_x = cx - nx0 * w_r
csv_right_y = cy - ny0 * w_r

csv_width = np.sqrt(
    (csv_left_x - csv_right_x)**2 +
    (csv_left_y - csv_right_y)**2
)

csv_w_tr_left_m = abs(np.sqrt((cx - csv_left_x)**2 + (cy - csv_left_y)**2))
csv_w_tr_right_m = abs(np.sqrt((cx - csv_right_x)**2 + (cy - csv_right_y)**2))

# ─── 境界座標をセンターライン法線への射影距離（正の幅）に変換 ────────────
# visualize_track.py が使う単純ロール法線（左向き）で再射影する
dx_roll = np.roll(cx, -1) - np.roll(cx, 1)
dy_roll = np.roll(cy, -1) - np.roll(cy, 1)
norm_roll = np.sqrt(dx_roll**2 + dy_roll**2) + 1e-8
lnx = -dy_roll / norm_roll   # 左向き法線
lny =  dx_roll / norm_roll

new_w_l = np.zeros(N)
new_w_r = np.zeros(N)
for i in range(N):
    P   = np.array([cx[i], cy[i]])
    n_l = np.array([lnx[i], lny[i]])  # 左向き法線
    left_pt  = np.array([csv_left_x[i],  csv_left_y[i]])
    right_pt = np.array([csv_right_x[i], csv_right_y[i]])
    # 左境界: 左向き法線への射影の絶対値
    new_w_l[i] = abs(np.dot(left_pt  - P, n_l))
    # 右境界: 右方向(=-左法線)への射影の絶対値
    new_w_r[i] = abs(np.dot(right_pt - P, n_l))

# 異常値クリップ
new_w_l = np.clip(new_w_l,  0.3, 8.0)
new_w_r = np.clip(new_w_r,  0.3, 8.0)

print(f"new_w_l: min={new_w_l.min():.3f}  max={new_w_l.max():.3f}")
print(f"new_w_r: min={new_w_r.min():.3f}  max={new_w_r.max():.3f}")

# ─── track_revised.csv として保存 ─────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
    for i in range(N):
        writer.writerow([cx[i], cy[i], new_w_r[i], new_w_l[i]])
print(f"Saved: {OUT_CSV}  ({N} rows)")

# ─── 左右境界座標を1つの CSV にまとめて保存 ──────────────────────────────
# フォーマット: left_x, left_y, right_x, right_y (各行が同一ウェイポイントに対応)
OUT_BOUNDARY = os.path.join(_dir, "../env/boundary.csv")

with open(OUT_BOUNDARY, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["left_x", "left_y", "right_x", "right_y", "center_x", "center_y","width","w_tr_left_m","w_tr_right_m"])
    for i in range(N):
        writer.writerow([csv_left_x[i], csv_left_y[i],
                         csv_right_x[i], csv_right_y[i],
                         cx[i], cy[i],csv_width[i],csv_w_tr_left_m[i],csv_w_tr_right_m[i]])
print(f"Saved: {OUT_BOUNDARY}  ({N} rows)  ← original left & right boundaries + centerline")

# ─── boundary.csv を読み込んで可視化 ──────────────────────────────────────
boundary = np.loadtxt(OUT_BOUNDARY, delimiter=",", skiprows=1)
b_lx = boundary[:, 0]
b_ly = boundary[:, 1]
b_rx = boundary[:, 2]
b_ry = boundary[:, 3]

plt.figure(figsize=(10, 10))
plt.plot(b_lx, b_ly, 'r', linewidth=2, label="boundary left")
plt.plot(b_rx, b_ry, 'b', linewidth=2, label="boundary right")
plt.plot(cx, cy, 'k--', linewidth=1.5, label="centerline")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.title("boundary.csv  (left_x, left_y, right_x, right_y)")
OUT_PNG2 = os.path.join(_dir, "../env/plot_boundary.png")
plt.savefig(OUT_PNG2, dpi=300)
print(f"Saved plot: {OUT_PNG2}")