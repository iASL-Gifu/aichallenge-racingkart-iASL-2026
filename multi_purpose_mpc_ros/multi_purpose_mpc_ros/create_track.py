import lanelet2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

def remove_duplicate_points(points, eps=1e-3):
    cleaned = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - cleaned[-1]) > eps:
            cleaned.append(p)
    return np.array(cleaned)

##################################
# 法線とpolylineの交点を探す
##################################
def intersect_normal_with_polyline(P, n, polyline):

    best_abs_dist = 1e9
    best_point = None
    best_d = None

    A = P - 20*n
    B = P + 20*n

    for i in range(len(polyline)-1):

        C = polyline[i]
        D = polyline[i+1]

        M = np.array([
            [B[0]-A[0], C[0]-D[0]],
            [B[1]-A[1], C[1]-D[1]]
        ])

        rhs = np.array([
            C[0]-A[0],
            C[1]-A[1]
        ])

        det = np.linalg.det(M)

        if abs(det) < 1e-8:
            continue

        t, u = np.linalg.solve(M, rhs)

        if 0 <= t <= 1 and 0 <= u <= 1:

            X = A + t*(B-A)

            d = np.dot(X-P, n)

            if abs(d) < best_abs_dist:
                best_abs_dist = abs(d)
                best_point = X
                best_d = d

    return best_point, best_d
##################################
# Centerlineを等間隔にリサンプル
##################################
def resample_polyline_with_width(points, widths, ds=10.0):

    d = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0)

    s_new = np.arange(0, s[-1], ds)

    x_new = np.interp(s_new, s, points[:,0])
    y_new = np.interp(s_new, s, points[:,1])
    w_new = np.interp(s_new, s, widths)

    return np.column_stack((x_new,y_new)), w_new

##################################
# map読み込み
##################################

map_file = "../../aichallenge_submit_launch/map/lanelet2_map.osm"
projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.625, 139.781))
lanelet_map = lanelet2.io.load(map_file, projector)
lanelets = list(lanelet_map.laneletLayer)

print("num lanelets =", len(lanelets))



##################################
# centerline取得
##################################



segments = []


# UtmProjector(useOffset=True) は Origin からの相対座標を返す（例: p.x ≈ 20 m）
# origin_x/y を「足す」ことで OSM の local_x/local_y と同じ絶対 UTM 座標に変換する
for ll in lanelets:

    origin_x = 89608.61776988552
    origin_y = 43116.542553572915

    center = np.array([
        [p.x + origin_x, p.y + origin_y]
        for p in ll.centerline
    ])

    left = np.array([
        [p.x + origin_x, p.y + origin_y]
        for p in ll.leftBound
    ])

    right = np.array([
        [p.x + origin_x, p.y + origin_y]
        for p in ll.rightBound
    ])

    n = min(len(left), len(right))

    width = np.mean(
        np.linalg.norm(left[:n]-right[:n], axis=1)
    )

    segments.append({
        "id": ll.id,
        "center": center,
        "left": left,
        "right": right,
        "width": width
    })

id_to_seg = {}
for seg in segments:
    id_to_seg[seg["id"]] = seg

track_ids = [
    9169,
    14,
    9163,
    9157,
    9151,
    2263,
    2256,
    9145,
    9134,
    9128,
    9123,
    9118,
    1483
]

# 9169を反転
id_to_seg[9169]["center"] = id_to_seg[9169]["center"][::-1]
id_to_seg[9169]["left"]   = id_to_seg[9169]["left"][::-1]
id_to_seg[9169]["right"]  = id_to_seg[9169]["right"][::-1]

ordered = []

for id_ in track_ids:

    seg = id_to_seg[id_].copy()

    if len(ordered) != 0:

        prev_end = ordered[-1]["center"][-1]

        start = seg["center"][0]
        end = seg["center"][-1]

        d_start = np.linalg.norm(prev_end-start)
        d_end = np.linalg.norm(prev_end-end)

        if d_end < d_start:
            seg["center"] = seg["center"][::-1]
            seg["left"] = seg["left"][::-1]
            seg["right"] = seg["right"][::-1]
    ordered.append(seg)

for id_ in [9169,14,1483]:
    seg=id_to_seg[id_]

    print("id=",id_)
    print("start=",seg["center"][0])
    print("end  =",seg["center"][-1])
    print()



left_all = []
right_all = []

for seg in ordered:
    left_all.extend(seg["left"])
    right_all.extend(seg["right"])

for i in range(len(ordered)):
    cur = ordered[i]
    nxt = ordered[(i+1)%len(ordered)]

    cur_end = cur["center"][-1]
    nxt_start = nxt["center"][0]

    d = np.linalg.norm(cur_end - nxt_start)
    d = max(d, 1e-6)
    s = np.insert(np.cumsum(d), 0, 0.0)

    print(cur["id"], "->", nxt["id"], "distance =", d)

left_all = np.vstack([left_all, left_all[0]])
right_all = np.vstack([right_all, right_all[0]])



print(left_all.shape)
print(right_all.shape)
'''
print("ordered ids:")
for s in ordered:
    print(s["id"])
'''
##################################
# centerline連結
##################################

points = []
for seg in ordered:

    c = seg["center"]

    if len(points) != 0:
        c = c[1:]

    points.extend(c)

points = np.array(points)


points = remove_duplicate_points(points)

# 閉ループ化
if np.linalg.norm(points[0] - points[-1]) > 1e-6:
    points = np.vstack([points, points[0]])

# centerlineを0.5m間隔にリサンプル
d = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
s = np.insert(np.cumsum(d), 0, 0.0)

# normalize
eps = 1e-6
s = np.maximum.accumulate(s)  # 単調保証
s = s + np.arange(len(s)) * eps
s = s / s[-1]

# periodic spline
cs_x = CubicSpline(s, points[:,0], bc_type="periodic")
cs_y = CubicSpline(s, points[:,1], bc_type="periodic")

s_new = np.linspace(0, 1, 4000)
x_new = cs_x(s_new)
y_new = cs_y(s_new)


points = np.column_stack((x_new, y_new))

'''
x_s = gaussian_filter1d(x, sigma=3, mode='wrap')
y_s = gaussian_filter1d(y, sigma=3, mode='wrap')

s = np.zeros(len(x))
s[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

cs_x = CubicSpline(s, x, bc_type='periodic')
cs_y = CubicSpline(s, y, bc_type='periodic')
'''
#dx0 = np.gradient(x_new)
#dy0 = np.gradient(y_new)
dx0 = gaussian_filter1d(x_new, sigma=4, order=1, mode='wrap')
dy0 = gaussian_filter1d(y_new, sigma=4, order=1, mode='wrap')

norm0 = np.sqrt(dx0**2+dy0**2)+ 1e-8

nx0 = dy0/norm0
ny0 = -dx0/norm0

w_left = []
w_right = []

left_hit_points = []
right_hit_points = []

left_hit_points = []
right_hit_points = []

for i in range(len(points)):

    P = points[i]
    n = np.array([nx0[i], ny0[i]])

    hit_left, d_left = intersect_normal_with_polyline(
        P,
        n,
        left_all
    )

    hit_right, d_right = intersect_normal_with_polyline(
        P,
        -n,
        right_all
    )

    # 左側
    if hit_left is None:
        d_left = w_left[-1]

        if len(left_hit_points) > 0:
            hit_left = P+n*d_left
        else:
            hit_left = P + n*d_left

    # 右側
    if hit_right is None:
        d_right = w_right[-1]

        if len(right_hit_points) > 0:
            hit_right = P-n*d_right
        else:
            hit_right = P - n*d_right

    w_left.append(d_left)
    w_right.append(d_right)

    left_hit_points.append(hit_left)
    right_hit_points.append(hit_right)

    if np.dot(hit_left - P, n) < 0:
        continue

    test = np.dot(hit_left - P, n)

    print(
        "norm =", d_left,
        "signed =", test
    )

w_left = np.array(w_left)
w_right = np.array(w_right)

#w_left = np.clip(w_left,0.3,5.0)
#w_right = np.clip(w_right,0.3,5.0)


import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_out_track_path = os.path.join(_script_dir, "../env/track_center.csv")

with open(_out_track_path,"w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "x_m",
        "y_m",
        "w_tr_right_m",
        "w_tr_left_m"
    ])

    for i in range(len(points)):

        writer.writerow([
            points[i,0],
            points[i,1],
            abs(w_right[i]),
            abs(w_left[i])
        ])

track = np.loadtxt(_out_track_path, delimiter=",", skiprows=1)

x = track[:,0]
y = track[:,1]
wr = track[:,2]
wl = track[:,3]

print("track_center.csv saved")

##################################
# waypoint_bounds.csv 生成
# MPC waypoints (traj_race_cl_mpc.csv) の各点に対して、
# lanelet2 の左右境界ポリライン (left_all / right_all) への
# 法線方向交点距離 (ub, lb) を直接計算して保存する。
# reference_path.py はこの CSV を読み込んで wp.ub / wp.lb に直接代入するだけなので
# track.csv の射影誤差が完全に排除される。
##################################

import os

# waypoint CSV のパスを自動解決
_script_dir = os.path.dirname(os.path.abspath(__file__))
_wp_candidates = [
    os.path.join(_script_dir, "../env/min_curv/traj_race_cl_mpc.csv"),
    #os.path.join(_script_dir, "../../global_racetrajectory_optimization/outputs/traj_race_cl_mpc.csv"),
]
_wp_path = None
for _c in _wp_candidates:
    if os.path.exists(_c):
        _wp_path = _c
        break

if _wp_path is None:
    print("[waypoint_bounds] WARNING: traj_race_cl_mpc.csv not found. Skipping.")
else:
    print(f"[waypoint_bounds] Loading waypoints from: {_wp_path}")
    _wp_data = np.loadtxt(_wp_path, delimiter=",", skiprows=1)
    # 列: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps3
    _wp_x   = _wp_data[:, 1]
    _wp_y   = _wp_data[:, 2]
    _wp_psi = _wp_data[:, 3]

    # MPC coordinate offset (reference_path.py と同じ値)
    _X_OFFSET = 5.332886
    _Y_OFFSET = -75.727413
    _wp_x = _wp_x + _X_OFFSET
    _wp_y = _wp_y + _Y_OFFSET

    _ub_list = []
    _lb_list = []

    for _i in range(len(_wp_x)):
        _P   = np.array([_wp_x[_i], _wp_y[_i]])
        _psi = _wp_psi[_i]
        # waypoint の左法線 (psi + pi/2 方向)
        _n = np.array([-np.sin(_psi), -np.cos(_psi)])

        # 左壁 (left_all) への法線交点 → ub
        _hit_l, _d_l = intersect_normal_with_polyline(_P,  _n, left_all)
        # 右壁 (right_all) への法線交点 → lb (負符号)
        _hit_r, _d_r = intersect_normal_with_polyline(_P, -_n, right_all)

        # フォールバック: 交点が見つからない場合は前の値を使う
        if _d_l is None or not np.isfinite(_d_l):
            _d_l = _ub_list[-1] if _ub_list else 3.5
        if _d_r is None or not np.isfinite(_d_r):
            _d_r = abs(_lb_list[-1]) if _lb_list else 3.5

        _ub_list.append(float(_d_l))
        _lb_list.append(-float(_d_r))

    _ub_arr = np.array(_ub_list)
    _lb_arr = np.array(_lb_list)

    # 外れ値クリップ: コース幅は最大 6m 程度
    _ub_arr = np.clip(_ub_arr, 0.5, 6.0)
    _lb_arr = np.clip(_lb_arr, -6.0, -0.5)

    # 出力先を env フォルダに合わせる
    _out_path = os.path.join(_script_dir, "../env/waypoint_bounds.csv")
    with open(_out_path, "w", newline="") as _f:
        _writer = csv.writer(_f)
        _writer.writerow(["idx", "ub", "lb"])
        for _i in range(len(_ub_arr)):
            _writer.writerow([_i, _ub_arr[_i], _lb_arr[_i]])

    print(f"[waypoint_bounds] Saved {len(_ub_arr)} entries to: {_out_path}")
    print(f"[waypoint_bounds] ub: min={_ub_arr.min():.3f}, max={_ub_arr.max():.3f}")
    print(f"[waypoint_bounds] lb: min={_lb_arr.min():.3f}, max={_lb_arr.max():.3f}")


if np.linalg.norm([x[-1]-x[0], y[-1]-y[0]]) > 1e-9:
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    wr = np.append(wr, wr[0])
    wl = np.append(wl, wl[0])


plt.figure(figsize=(8,8))

# 元境界
plt.plot(left_all[:,0],left_all[:,1],'r--',label='true left')
plt.plot(right_all[:,0],right_all[:,1],'b--',label='true right')

# csvから復元した境界
left_hit_points=np.array(left_hit_points)
right_hit_points=np.array(right_hit_points)

# csvから復元するための法線

csv_left_x  = points[:,0] + nx0*w_left
csv_left_y  = points[:,1] + ny0*w_left

csv_right_x = points[:,0] - nx0*w_right
csv_right_y = points[:,1] - ny0*w_right

error = np.linalg.norm(
    left_hit_points -
    (points + np.column_stack((nx0,ny0))*w_left[:,None]),
    axis=1
)

print(error.max())

p = lanelets[0].centerline[0]
print(p.x, p.y)


left_hit_points = np.array(left_hit_points)
right_hit_points = np.array(right_hit_points)

plt.figure(figsize=(8,8))

# 元境界
plt.plot(left_all[:,0], left_all[:,1],
         'r--', linewidth=2, label="true left")

plt.plot(right_all[:,0], right_all[:,1],
         'b--', linewidth=2, label="true right")

# 法線交点
plt.plot(left_hit_points[:,0], left_hit_points[:,1],
         'm', linewidth=1.5, label="intersection left")

plt.plot(right_hit_points[:,0], right_hit_points[:,1],
         'c', linewidth=1.5, label="intersection right")

# csvから復元した境界
plt.plot(csv_left_x, csv_left_y,
         'r', linewidth=2, label="csv left")

plt.plot(csv_right_x, csv_right_y,
         'b', linewidth=2, label="csv right")

# centerline
plt.plot(x, y, 'k', linewidth=2, label="center")

plt.axis("equal")
plt.legend()
plt.show()
'''

plt.axis("equal")
plt.show()
plt.plot(x,y,'k',label='center')

plt.axis("equal")
plt.legend()
plt.show()
'''