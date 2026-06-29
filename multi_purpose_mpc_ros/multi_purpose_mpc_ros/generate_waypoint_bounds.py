"""
generate_waypoint_bounds.py
===========================
track.csv と traj_race_cl_mpc.csv から waypoint_bounds.csv を生成する。

アルゴリズム:
  各 waypoint から左壁・右壁ポリラインの最近傍点を cKDTree で高速探索し、
  waypoint の法線方向への符号付き射影距離として ub/lb を計算する。

使い方:
  python3 generate_waypoint_bounds.py

出力:
  ../env/waypoint_bounds.csv  (idx, ub, lb)
"""

import numpy as np
import csv
import os
import math
from scipy.spatial import cKDTree

_dir = os.path.dirname(os.path.abspath(__file__))
TRACK_CSV = os.path.join(_dir, "../env/track.csv")
WP_CSV    = os.path.join(_dir, "../env/min_curv/traj_race_cl_mpc.csv")
OUT_CSV   = os.path.join(_dir, "../env/waypoint_bounds.csv")

# reference_path.py と同じ座標オフセット
X_OFFSET = 5.332886
Y_OFFSET = -75.727413

# =====================================================================
# track.csv から左右壁ポリラインを再構成
# =====================================================================
print(f"Loading track.csv: {TRACK_CSV}")
track = np.loadtxt(TRACK_CSV, delimiter=",", skiprows=1)

# track.csv にオフセットを加算して waypoint の raw 座標系に合わせる
t_x  = track[:, 0] + X_OFFSET
t_y  = track[:, 1] + Y_OFFSET
t_wr = track[:, 2]   # right half-width
t_wl = track[:, 3]   # left  half-width

# センターライン各点の進行方向（前後差分で循環計算）
dx = np.roll(t_x, -1) - np.roll(t_x, 1)
dy = np.roll(t_y, -1) - np.roll(t_y, 1)
norm_d = np.sqrt(dx**2 + dy**2) + 1e-8
# 左法線ベクトル (進行方向の +90°)
lnx = -dy / norm_d
lny =  dx / norm_d

# 絶対座標の左右壁
left_wall  = np.column_stack([t_x + lnx * t_wl, t_y + lny * t_wl])
right_wall = np.column_stack([t_x - lnx * t_wr, t_y - lny * t_wr])

print(f"  Track points : {len(t_x)}")
print(f"  Left wall x  : [{left_wall[:,0].min():.1f}, {left_wall[:,0].max():.1f}]")
print(f"  Left wall y  : [{left_wall[:,1].min():.1f}, {left_wall[:,1].max():.1f}]")

left_tree  = cKDTree(left_wall)
right_tree = cKDTree(right_wall)

# =====================================================================
# MPC waypoints のロード
# =====================================================================
print(f"\nLoading waypoints: {WP_CSV}")
wp_data = np.loadtxt(WP_CSV, delimiter=",", skiprows=1)
# 列: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps3
# waypoints の raw 座標をそのまま使用（オフセットなし）
wp_x   = wp_data[:, 1]
wp_y   = wp_data[:, 2]
wp_psi = wp_data[:, 3]
N = len(wp_x)
print(f"  Waypoints    : {N}")
print(f"  WP x range   : [{wp_x.min():.1f}, {wp_x.max():.1f}]")
print(f"  WP y range   : [{wp_y.min():.1f}, {wp_y.max():.1f}]")

# =====================================================================
# 各 waypoint で左右壁への符号付き距離を計算
# アルゴリズム:
#   1. KDTree で k近傍点（k=10）を取得
#   2. waypoint 法線方向 (n_left) への射影値 = signed distance
#   3. 正の最大値 → ub (左側), 負の最小値 → lb (右側)
# =====================================================================
wp_pts = np.column_stack([wp_x, wp_y])
K = 30  # 近傍点数（多いほど遠い壁も捕捉しやすい）

_, left_idx  = left_tree.query(wp_pts,  k=K)
_, right_idx = right_tree.query(wp_pts, k=K)

ub_list = []
lb_list = []

for i in range(N):
    P   = np.array([wp_x[i], wp_y[i]])
    psi = wp_psi[i]
    n_l = np.array([-math.sin(psi), math.cos(psi)])   # 左法線

    # 左壁: k近傍の中で射影値が最も正（左側）の点を採用
    d_l = 0.5  # fallback
    for j in left_idx[i]:
        v = float(np.dot(left_wall[j] - P, n_l))
        if v > d_l:
            d_l = v

    # 右壁: k近傍の中で射影値が最も負（右側）の点を採用
    d_r = -0.5  # fallback
    for j in right_idx[i]:
        v = float(np.dot(right_wall[j] - P, n_l))
        if v < d_r:
            d_r = v

    ub_list.append(d_l)
    lb_list.append(d_r)

ub_arr = np.array(ub_list)
lb_arr = np.array(lb_list)

# 外れ値クリップ
ub_arr = np.clip(ub_arr,  0.3, 8.0)
lb_arr = np.clip(lb_arr, -8.0, -0.3)

print(f"\nub: min={ub_arr.min():.3f}  max={ub_arr.max():.3f}  mean={ub_arr.mean():.3f}")
print(f"lb: min={lb_arr.min():.3f}  max={lb_arr.max():.3f}  mean={lb_arr.mean():.3f}")

# =====================================================================
# 保存
# =====================================================================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "ub", "lb"])
    for i in range(N):
        writer.writerow([i, ub_arr[i], lb_arr[i]])

print(f"\nSaved: {OUT_CSV}  ({N} rows)")
