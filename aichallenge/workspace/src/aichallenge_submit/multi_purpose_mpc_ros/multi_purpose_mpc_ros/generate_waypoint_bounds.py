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
TRACK_CSV = os.path.join(_dir, "boundary_true.csv")
WP_CSV    = os.path.join(_dir, "../env/centerline/traj_center313.csv")
OUT_CSV   = os.path.join(_dir, "../env/centerline/waypoint_bounds_center.csv")

count = 0

# reference_path.py と同じ座標オフセット
X_OFFSET = 5.332886
Y_OFFSET = -75.727413

# =====================================================================
# boundary.csv から左右境界座標を読み込む
# =====================================================================
print(f"Loading boundary.csv: {TRACK_CSV}")
track = np.loadtxt(TRACK_CSV, delimiter=",", skiprows=1)

# track.csv にオフセットを加算して waypoint の raw 座標系に合わせる
#x  = track[:, 4] #+ X_OFFSET
#y  = track[:, 5] #+ Y_OFFSET
#t_wr = track[:, 8]   # right half-width
#t_wl = track[:, 7]   # left  half-width

# センターライン各点の進行方向（前後差分で循環計算）
#dx = np.roll(t_x, -1) - np.roll(t_x, 1)
#dy = np.roll(t_y, -1) - np.roll(t_y, 1)
#norm_d = np.sqrt(dx**2 + dy**2) + 1e-8
# 左法線ベクトル (進行方向の +90°)
#lnx = -dy / norm_d
#lny =  dx / norm_d

# 絶対座標の左右壁
left_wall=np.column_stack([
track[:,0]+ X_OFFSET,
track[:,1]+ Y_OFFSET
])

right_wall=np.column_stack([
track[:,2]+ X_OFFSET,
track[:,3]+ Y_OFFSET
])

print(f"Track points : {len(left_wall)}")
print(f"  Left wall x  : [{left_wall[:,0].min():.1f}, {left_wall[:,0].max():.1f}]")
print(f"  Left wall y  : [{left_wall[:,1].min():.1f}, {left_wall[:,1].max():.1f}]")

# =====================================================================
# MPC waypoints のロード
# =====================================================================
print(f"\nLoading waypoints: {WP_CSV}")

wp_data = np.loadtxt(WP_CSV, delimiter=",", skiprows=1)
print("wp_data.shape =", wp_data.shape)

print(open(WP_CSV).read().splitlines()[-3:])
# 列: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps3
# waypoints の raw 座標をそのまま使用（オフセットなし）
wp_x   = wp_data[:, 1]#+X_OFFSET
wp_y   = wp_data[:, 2]#+Y_OFFSET
wp_psi = wp_data[:,3]

import matplotlib.pyplot as plt

'''
# =====================================================
# センターラインからpsiを再計算
# =====================================================

dx = np.roll(wp_x, -1) - np.roll(wp_x, 1)
dy = np.roll(wp_y, -1) - np.roll(wp_y, 1)

length = np.hypot(dx, dy)
length[length < 1e-8] = 1e-8

tx = dx / length
ty = dy / length

# 接線方向→psi
#nx = ty
#ny = -tx

wp_psi = np.arctan2(-tx, ty)

# CSVデータも更新
wp_data[:,3] = wp_psi

NEW_WP_CSV = os.path.join(
    _dir,
    "../env/centerline/traj_center313_fixed.csv"
)

header = "s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps3"

np.savetxt(
    NEW_WP_CSV,
    wp_data,
    delimiter=",",
    fmt=[
        "%.0f",     # s_m
        "%.10f",    # x
        "%.10f",    # y
        "%.10f",    # psi
        "%.10f",    # kappa
        "%.10f",    # vx
        "%.10f"     # ax
    ],
    header=header,
    comments=""
)

print(f"Saved fixed waypoint csv : {NEW_WP_CSV}")

print("psi recalculated from centerline.")
'''


N = len(wp_x)
print(f"  Waypoints    : {N}")
print(f"  WP x range   : [{wp_x.min():.1f}, {wp_x.max():.1f}]")
print(f"  WP y range   : [{wp_y.min():.1f}, {wp_y.max():.1f}]")

# =====================================================================
# 各 waypoint で左右壁への符号付き距離を計算
# =====================================================================

ub_list = []
lb_list = []
boundary_idx_list = []
width_list = []
left_hit_list = []
right_hit_list = []

def intersect_normal_with_polyline(P, n, polyline):
    """
    P : waypoint
    n : 法線ベクトル（単位ベクトル）
    polyline : Nx2

    戻り値
      hit_point
      signed_distance
      segment_index
    """

    best_abs = np.inf
    best_point = None
    best_dist = None
    best_idx = -1

    # 十分長い法線
    A = P - 20.0 * n
    B = P + 20.0 * n

    for i in range(len(polyline)):

        C = polyline[i]
        D = polyline[(i+1) % len(polyline)]

        M = np.array([
            [B[0]-A[0], C[0]-D[0]],
            [B[1]-A[1], C[1]-D[1]]
        ])

        rhs = np.array([
            C[0]-A[0],
            C[1]-A[1]
        ])

        det = np.linalg.det(M)

        if abs(det) < 1e-10:
            continue

        t, u = np.linalg.solve(M, rhs)

        if 0 <= t <= 1 and 0 <= u <= 1:

            X = A + t * (B-A)

            d = np.dot(X-P, n)

            if abs(d) < best_abs:
                best_abs = abs(d)
                best_point = X
                best_dist = d
                best_idx = i

    return best_point, best_dist, best_idx



for i in range(N):
    P = np.array([wp_x[i], wp_y[i]])

    n = np.array([
        -math.cos(wp_psi[i]),
        -math.sin(wp_psi[i])
    ])

    hit_l, d_l, idx_l = intersect_normal_with_polyline(
        P,
        n,
        left_wall
    )
    #print(i, d_l, idx_l)

    hit_r, d_r, idx_r = intersect_normal_with_polyline(
        P,
        -n,
        right_wall
    )
    if hit_l is None:
        hit_l = np.array([np.nan, np.nan])

    if hit_r is None:
        hit_r = np.array([np.nan, np.nan])

    left_hit_list.append(hit_l)
    right_hit_list.append(hit_r)


    if idx_r == -1:
      import matplotlib.pyplot as plt

      plt.figure(figsize=(7,7))
      plt.plot(left_wall[:,0], left_wall[:,1])
      plt.plot(right_wall[:,0], right_wall[:,1])

      plt.scatter(P[0],P[1],c='r')

      A=P-100*n
      B=P+100*n

      plt.plot([A[0],B[0]],[A[1],B[1]],'r')

      plt.axis('equal')
      plt.show()
  


    if d_l is None:
        d_l = 8.0
        idx_l = -1

    if d_r is None:
        d_r = 8.0
        idx_r = -1

    ub_list.append(d_l)

    lb_list.append(-abs(d_r))
    #boundary_idx_list.append(idx_l)
    boundary_idx_list.append((idx_l, idx_r))
    width_list.append(abs(d_l+d_r))

ub_arr = np.array(ub_list)
lb_arr = np.array(lb_list)

# 外れ値クリップ
ub_arr = np.clip(ub_arr,  0.3, 8.0)
lb_arr = np.clip(lb_arr, -8.0, -0.3)

#dist, center_idx = center_tree.query(wp_pts)

#d = np.linalg.norm(center - wp_pts[0], axis=1)
#print(d.min())
#print("boundary center first =", center[0])
#print("waypoint first        =", wp_pts[0])
#print("max =", dist.max())
#print("mean =", dist.mean())
#print("min =", dist.min())

print(f"\nub: min={ub_arr.min():.3f}  max={ub_arr.max():.3f}  mean={ub_arr.mean():.3f}")
print(f"lb: min={lb_arr.min():.3f}  max={lb_arr.max():.3f}  mean={lb_arr.mean():.3f}")

# =====================================================================
# 保存
# =====================================================================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx",
        "ub", "lb",
        "left_boundary_idx",
        "right_boundary_idx",
        "left_x", "left_y",
        "right_x", "right_y"
        ])
    for i in range(N):
        writer.writerow([
                      i,
            ub_arr[i],
            lb_arr[i],
            boundary_idx_list[i][0],
            boundary_idx_list[i][1],
            left_hit_list[i][0], left_hit_list[i][1],
            right_hit_list[i][0], right_hit_list[i][1],
        ])

print(f"\nSaved: {OUT_CSV}  ({N} rows)")


# =====================================================================
# 可視化
# =====================================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

# 元の左右境界
plt.plot(left_wall[:,0], left_wall[:,1],
         'k-', linewidth=2, label='True Left Boundary')

plt.plot(right_wall[:,0], right_wall[:,1],
         'k-', linewidth=2, label='True Right Boundary')

# waypoint
plt.plot(wp_x, wp_y,
         'b-', linewidth=1.5, label='Center Line')

# 計算した交点
left_hit = np.array(left_hit_list)
right_hit = np.array(right_hit_list)

valid_left = ~np.isnan(left_hit[:,0])
valid_right = ~np.isnan(right_hit[:,0])

plt.scatter(left_hit[valid_left,0],
            left_hit[valid_left,1],
            s=10,
            c='red',
            label='Calculated Left')

plt.scatter(right_hit[valid_right,0],
            right_hit[valid_right,1],
            s=10,
            c='lime',
            label='Calculated Right')

# waypoint→交点
step = 20       # 全部描くと見づらいので20点おき

for i in range(0, N, step):

    if valid_left[i]:
        plt.plot(
            [wp_x[i], left_hit[i,0]],
            [wp_y[i], left_hit[i,1]],
            'r-',
            linewidth=0.7
        )

    if valid_right[i]:
        plt.plot(
            [wp_x[i], right_hit[i,0]],
            [wp_y[i], right_hit[i,1]],
            'g-',
            linewidth=0.7
        )

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Waypoint -> Boundary Intersection")
plt.show()