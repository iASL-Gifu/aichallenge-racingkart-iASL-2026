import os
import csv
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------
# 1. 既存の track.csv から境界線ポリラインを復元する
# ----------------------------------------------------
track_csv_path = "/home/takenoyama/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/track.csv"
print(f"Loading existing track.csv from: {track_csv_path}")

track_data = []
with open(track_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_data.append([
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["w_tr_right_m"]),
            float(row["w_tr_left_m"])
        ])
track_data = np.array(track_data)

tr_x = track_data[:, 0]
tr_y = track_data[:, 1]
tr_wr = track_data[:, 2]
tr_wl = track_data[:, 3]

# 各点での進行方向 psi を計算
dx_tr = gaussian_filter1d(tr_x, sigma=2, order=1, mode="wrap")
dy_tr = gaussian_filter1d(tr_y, sigma=2, order=1, mode="wrap")
psi_tr = np.arctan2(dy_tr, dx_tr)

# 左右境界線の座標を復元
left_all = []
right_all = []
for i in range(len(tr_x)):
    psi = psi_tr[i]
    n_x = -np.sin(psi)
    n_y = np.cos(psi)
    
    left_all.append([
        tr_x[i] + tr_wl[i] * n_x,
        tr_y[i] + tr_wl[i] * n_y
    ])
    right_all.append([
        tr_x[i] - tr_wr[i] * n_x,
        tr_y[i] - tr_wr[i] * n_y
    ])
left_all = np.array(left_all)
right_all = np.array(right_all)

# 閉ループ化の保証
if np.linalg.norm(left_all[0] - left_all[-1]) > 1e-3:
    left_all = np.vstack([left_all, left_all[0]])
if np.linalg.norm(right_all[0] - right_all[-1]) > 1e-3:
    right_all = np.vstack([right_all, right_all[0]])

# ----------------------------------------------------
# 2. 法線とポリラインの交点計算関数
# ----------------------------------------------------
def intersect_normal_with_polyline(P, n, polyline):
    best_abs_dist = 1e9
    best_point = None
    best_d = None
    A = P - 20 * n
    B = P + 20 * n
    for i in range(len(polyline) - 1):
        C = polyline[i]
        D = polyline[i + 1]
        M = np.array([
            [B[0] - A[0], C[0] - D[0]],
            [B[1] - A[1], C[1] - D[1]]
        ])
        rhs = np.array([C[0] - A[0], C[1] - A[1]])
        det = np.linalg.det(M)
        if abs(det) < 1e-8:
            continue
        try:
            t, u = np.linalg.solve(M, rhs)
            if 0 <= t <= 1 and 0 <= u <= 1:
                X = A + t * (B - A)
                d = np.dot(X - P, n)
                if abs(d) < best_abs_dist:
                    best_abs_dist = abs(d)
                    best_point = X
                    best_d = d
        except np.linalg.LinAlgError:
            continue
    return best_point, best_d

# ----------------------------------------------------
# 3. center.csv の読み込みとリサンプル
# ----------------------------------------------------
center_csv_path = "/home/takenoyama/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/aichallenge-trajectory-editor/csv/center.csv"
print(f"Loading center.csv from: {center_csv_path}")

raw_data = []
with open(center_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_data.append([
            float(row["x"]),
            float(row["y"]),
            float(row["speed"])
        ])
raw_data = np.array(raw_data)

# 閉ループ化の保証
if np.linalg.norm(raw_data[0, :2] - raw_data[-1, :2]) > 1e-3:
    raw_data = np.vstack([raw_data, raw_data[0]])

# 重複点の削除
cleaned = [raw_data[0]]
for row in raw_data[1:]:
    if np.linalg.norm(row[:2] - cleaned[-1][:2]) > 1e-3:
        cleaned.append(row)
raw_data = np.array(cleaned)

# 累積距離 s の計算
dists = np.sqrt(np.sum(np.diff(raw_data[:, :2], axis=0)**2, axis=1))
s_raw = np.insert(np.cumsum(dists), 0, 0.0)

# リサンプル (0.6m 解像度)
resolution = 0.6
total_length = s_raw[-1]
s_new = np.arange(0.0, total_length, resolution)

# CubicSpline 補間 (周期境界)
cs_x = CubicSpline(s_raw, raw_data[:, 0], bc_type="periodic")
cs_y = CubicSpline(s_raw, raw_data[:, 1], bc_type="periodic")
cs_v = CubicSpline(s_raw, raw_data[:, 2], bc_type="periodic")

x_new = cs_x(s_new)
y_new = cs_y(s_new)
v_new = cs_v(s_new)

# ----------------------------------------------------
# 4. psi と kappa の計算
# ----------------------------------------------------
dx = gaussian_filter1d(x_new, sigma=2, order=1, mode="wrap")
dy = gaussian_filter1d(y_new, sigma=2, order=1, mode="wrap")
ddx = gaussian_filter1d(x_new, sigma=2, order=2, mode="wrap")
ddy = gaussian_filter1d(y_new, sigma=2, order=2, mode="wrap")

psi_new = np.arctan2(dy, dx)
kappa_new = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5

# ----------------------------------------------------
# 5. 左右境界（ub, lb）の計算
# ----------------------------------------------------
ub_list = []
lb_list = []

# MPC coordinate offset (reference_path.py で適用されるオフセット)
X_OFFSET = 5.332886
Y_OFFSET = -75.727413

# offset を適用した x, y
x_new_offset = x_new + X_OFFSET
y_new_offset = y_new + Y_OFFSET

for i in range(len(x_new)):
    P = np.array([x_new_offset[i], y_new_offset[i]])
    psi = psi_new[i]
    # 法線ベクトル
    n = np.array([-np.sin(psi), np.cos(psi)])
    
    # 左壁への距離 (ub)
    _, d_l = intersect_normal_with_polyline(P, n, left_all)
    # 右壁への距離 (lb)
    _, d_r = intersect_normal_with_polyline(P, -n, right_all)
    
    if d_l is None or not np.isfinite(d_l):
        d_l = ub_list[-1] if ub_list else 3.5
    if d_r is None or not np.isfinite(d_r):
        d_r = abs(lb_list[-1]) if lb_list else 3.5
        
    ub_list.append(float(d_l))
    lb_list.append(-float(d_r))

ub_arr = np.clip(np.array(ub_list), 0.5, 6.0)
lb_arr = np.clip(np.array(lb_list), -6.0, -0.5)

# ----------------------------------------------------
# 6. 保存処理
# ----------------------------------------------------
# MPC 基準軌道保存先
traj_output_path = "/home/takenoyama/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/min_curv/traj_center_mpc.csv"
os.makedirs(os.path.dirname(traj_output_path), exist_ok=True)

with open(traj_output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"])
    accum_s = 0.0
    for i in range(len(x_new)):
        if i > 0:
            accum_s += np.sqrt((x_new[i] - x_new[i-1])**2 + (y_new[i] - y_new[i-1])**2)
        writer.writerow([
            accum_s,
            x_new[i],
            y_new[i],
            psi_new[i],
            kappa_new[i],
            v_new[i],
            0.0 # acceleration dummy
        ])
print(f"Saved MPC trajectory to: {traj_output_path}")

# 境界ファイル保存先
bounds_output_path = "/home/takenoyama/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/waypoint_bounds.csv"
with open(bounds_output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "ub", "lb"])
    for i in range(len(ub_arr)):
        writer.writerow([i, ub_arr[i], lb_arr[i]])
print(f"Saved waypoint bounds to: {bounds_output_path}")
