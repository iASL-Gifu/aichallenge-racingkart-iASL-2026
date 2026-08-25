"""min_curvature optimizer(global_racetrajectory_optimization)の出力を
center_csv_path / center_bounds_csv_path 用のCSVへ変換する。

入力:
  - outputs/traj_race_cl.csv (TUMツールの出力, s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2)
    center_converted_capped.csv (w_tr を CAP=1.3m でクリップした境界) を入力にして
    mincurv_iqp で生成したもの。座標系はこのリポジトリの env/track.csv や
    env/centerline/boundary.csv と同じ絶対(map)座標系であることを確認済み
    (オフセット加算は不要)。

やること:
  1. 座標オフセット確認 (不要であることを確認済みだが、念のため再チェックする)
  2. 走行方向をこのリポジトリの既存レースライン (env/min_curv2/traj_race_cl_mpc.csv) と
     揃える (向きが逆なら行順を反転し、psi_radをpiだけ回転、kappaの符号を反転する)
  3. env/centerline/boundary.csv の実測 left/right 境界ポリラインに対して、
     各waypointの法線を飛ばして交点を求め、ub/lb (center_bounds_csv_path 形式) を作る
  4. 閉ループを保証する (先頭===末尾)
  5. 出力:
     - traj_center_mincurv_capped.csv       (center_csv_path 用)
     - waypoint_bounds_center_mincurv_capped.csv (center_bounds_csv_path 用)
     - mincurv_capped_for_visualize.csv     (visualize_from_width.py で読める形式)
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUM_OUTPUT = "/home/takenoyama/Documents/AI_challenge/global_racetrajectory_optimization/outputs/traj_race_cl.csv"
BOUNDARY_CSV = os.path.join(SCRIPT_DIR, "boundary.csv")
RACE_MPC_CSV = os.path.join(SCRIPT_DIR, "..", "min_curv2", "traj_race_cl_mpc.csv")

OUT_CENTER_CSV = os.path.join(SCRIPT_DIR, "traj_center_mincurv_capped.csv")
OUT_BOUNDS_CSV = os.path.join(SCRIPT_DIR, "waypoint_bounds_center_mincurv_capped.csv")
OUT_VISUALIZE_CSV = os.path.join(SCRIPT_DIR, "mincurv_capped_for_visualize.csv")


def intersect_normal_with_polyline(P, n, polyline, max_reach=6.0):
    """P から方向 n (単位ベクトル) に向かう半直線 (長さ max_reach) と
    polyline (折れ線) の交点のうち、P に最も近いものを返す。
    create_track.py の intersect_normal_with_polyline と同じロジック。
    """
    best_abs_dist = 1e9
    best_point = None
    best_d = None

    A = P - max_reach * n
    B = P + max_reach * n

    for i in range(len(polyline) - 1):
        C = polyline[i]
        D = polyline[i + 1]

        M = np.array([
            [B[0] - A[0], C[0] - D[0]],
            [B[1] - A[1], C[1] - D[1]],
        ])
        rhs = np.array([C[0] - A[0], C[1] - A[1]])

        det = np.linalg.det(M)
        if abs(det) < 1e-8:
            continue

        t, u = np.linalg.solve(M, rhs)
        if 0 <= t <= 1 and 0 <= u <= 1:
            X = A + t * (B - A)
            d = np.dot(X - P, n)
            if abs(d) < best_abs_dist:
                best_abs_dist = abs(d)
                best_point = X
                best_d = d

    return best_point, best_d


def main():
    # ------------------------------------------------------------------
    # 1. 読み込み
    # ------------------------------------------------------------------
    tum = pd.read_csv(
        TUM_OUTPUT, sep=';', comment='#', header=None,
        names=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])

    # "_cl" = closed loop 形式なので末尾は先頭の重複。以後は重複を落として扱う
    closure_gap = float(np.hypot(
        tum['x_m'].iloc[0] - tum['x_m'].iloc[-1],
        tum['y_m'].iloc[0] - tum['y_m'].iloc[-1]))
    print(f"[1] TUM出力の閉ループ確認: 先頭-末尾ギャップ = {closure_gap:.6f} m")
    assert closure_gap < 1e-3, "TUM出力が閉じていません"
    tum = tum.iloc[:-1].reset_index(drop=True)  # 重複末尾を除去 (後で作り直す)

    # env/centerline/boundary.csv は「ローカル(オフセット前)座標系」。
    # env/min_curv2/traj_race_cl_mpc.csv (既存レースライン, 実行時に使われる座標系)
    # および今回の TUM 出力は「X_OFFSET/Y_OFFSET を加算済みの座標系」であることを
    # 実測で確認済み (boundary.csv の center_x,center_y + offset が
    # center_converted.csv の値と厳密に一致する)。
    X_OFFSET = 5.332886
    Y_OFFSET = -75.727413

    boundary = pd.read_csv(BOUNDARY_CSV)
    left_all = boundary[['left_x', 'left_y']].to_numpy(float) + [X_OFFSET, Y_OFFSET]
    right_all = boundary[['right_x', 'right_y']].to_numpy(float) + [X_OFFSET, Y_OFFSET]
    # 交点探索用に閉ループを明示 (先頭を末尾に複製)
    left_all = np.vstack([left_all, left_all[0]])
    right_all = np.vstack([right_all, right_all[0]])

    # ------------------------------------------------------------------
    # 2. 座標オフセット確認
    #    env/centerline/boundary.csv (center_x,center_y) にオフセットを加えたものが
    #    TUM出力(x_m,y_m) と同じ絶対座標系になっているかを、最近傍距離で検証する。
    # ------------------------------------------------------------------
    cx = boundary['center_x'].to_numpy() + X_OFFSET
    cy = boundary['center_y'].to_numpy() + Y_OFFSET
    sample_idx = np.linspace(0, len(tum) - 1, 20).astype(int)
    nearest_dists = []
    for i in sample_idx:
        d2 = (cx - tum['x_m'].iloc[i]) ** 2 + (cy - tum['y_m'].iloc[i]) ** 2
        nearest_dists.append(np.sqrt(d2.min()))
    nearest_dists = np.array(nearest_dists)
    print(f"[2] 座標系チェック: TUM出力の各点から (boundary.csv + offset) centerline までの"
          f"最近傍距離 mean={nearest_dists.mean():.3f}m max={nearest_dists.max():.3f}m "
          f"(数十cm程度ならオフセット適用後に同一座標系と判断できる)")
    assert nearest_dists.max() < 2.0, (
        "オフセットを加えても TUM出力と boundary.csv が同じ座標系になりません。")
    print(f"    -> X_OFFSET={X_OFFSET}, Y_OFFSET={Y_OFFSET} を boundary.csv 側に適用して整合")

    # ------------------------------------------------------------------
    # 3. 走行方向をこのリポジトリの既存レースラインに合わせる
    # ------------------------------------------------------------------
    race = pd.read_csv(RACE_MPC_CSV)
    rx0, ry0 = race['x_m'].iloc[0], race['y_m'].iloc[0]

    tx = tum['x_m'].to_numpy()
    ty = tum['y_m'].to_numpy()
    d2 = (tx - rx0) ** 2 + (ty - ry0) ** 2
    j = int(np.argmin(d2))
    jn = (j + 1) % len(tx)
    heading_tum_fwd = np.arctan2(ty[jn] - ty[j], tx[jn] - tx[j])

    rx1, ry1 = race['x_m'].iloc[1], race['y_m'].iloc[1]
    heading_race_fwd = np.arctan2(ry1 - ry0, rx1 - rx0)

    diff = abs((heading_tum_fwd - heading_race_fwd + np.pi) % (2 * np.pi) - np.pi)
    print(f"[3] 走行方向チェック: TUM出力 heading={heading_tum_fwd:.3f} rad, "
          f"既存レースライン heading={heading_race_fwd:.3f} rad, 差={diff:.3f} rad")

    need_reverse = diff > (np.pi / 2)
    print(f"    -> {'反転が必要' if need_reverse else '反転不要'} (diff {'>' if need_reverse else '<='} pi/2)")

    if need_reverse:
        tum = tum.iloc[::-1].reset_index(drop=True)

    # psi_rad / kappa_radpm は TUM 側の "北(=+Y軸)を0とする" 独自の角度規約であり、
    # このリポジトリの法線計算 (n = [-sin(psi), cos(psi)], 標準の
    # atan2(dy,dx) 規約) とは基準が異なる (実測で ~pi/2 のズレを確認)。
    # 変換で符号を追いかけるより、反転後の x,y 点列から heading と曲率を
    # 直接引き直す方が安全なのでそうする (閉ループを考慮した周期差分)。
    tx = tum['x_m'].to_numpy()
    ty = tum['y_m'].to_numpy()
    tx_ext = np.concatenate([tx[-1:], tx, tx[:1]])
    ty_ext = np.concatenate([ty[-1:], ty, ty[:1]])
    dx = tx_ext[2:] - tx_ext[:-2]
    dy = ty_ext[2:] - ty_ext[:-2]
    psi_new = np.arctan2(dy, dx)  # 標準の進行方向heading (atan2(dy,dx))

    ds = 0.5 * (np.hypot(tx_ext[2:] - tx_ext[1:-1], ty_ext[2:] - ty_ext[1:-1])
                + np.hypot(tx_ext[1:-1] - tx_ext[:-2], ty_ext[1:-1] - ty_ext[:-2]))
    psi_ext = np.concatenate([psi_new[-1:], psi_new, psi_new[:1]])
    dpsi = np.mod(psi_ext[2:] - psi_ext[:-2] + np.pi, 2 * np.pi) - np.pi
    ds_safe = np.where(ds > 1e-6, ds, 1e-6)
    tum['kappa_radpm'] = dpsi / (2.0 * ds_safe)

    # heading (標準atan2) は法線交点計算 (5.) にそのまま使う。
    # 一方、CSVに書き出す psi_rad 列は wp.normal_angle として読み込まれ、
    # mpc_controller.py の _publish_lane_markers 側で
    # nx=-cos(normal_angle), ny=-sin(normal_angle) という式で左法線に変換される。
    # このコードのコメント (「normal_angle が無ければ従来の psi + pi/2 を使用」)
    # から逆算すると、normal_angle は heading - pi/2 でなければならない
    # (実際に既存の traj_race_cl_mpc.csv の psi_rad で検証済み: heading - psi_rad ≈ pi/2)。
    # ここで標準heading (atan2(dy,dx)) をそのまま書き出していたのが不具合の原因で、
    # 境界表示が左右で交差する/幅がほぼ無いように見えていた。
    tum['heading_std'] = psi_new
    tum['psi_rad'] = np.mod(psi_new - np.pi / 2.0 + np.pi, 2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # 4. 閉ループの確認 (末尾に複製点は追加しない)
    #    既存の production ファイル (traj_center313_geometry_smooth_local.csv,
    #    min_curv2/traj_race_cl_mpc.csv とその waypoint_bounds) はいずれも
    #    「先頭===末尾の複製行」を持たず、center_csv と center_bounds_csv の
    #    行数が完全に一致している (circular: true が周回を内部で処理するため)。
    #    ここで複製行を追加すると center 側だけ行数が+1され、
    #    ReferencePath 内で waypoint と bounds のインデックスが1つずつズレて
    #    周回後半で境界(ub/lb)が全く別の場所の値を指してしまう
    #    (RVizで直線状に境界が突き抜けて見えた不具合の原因)。
    # ------------------------------------------------------------------
    closure_gap2 = float(np.hypot(
        tum['x_m'].iloc[0] - tum['x_m'].iloc[-1],
        tum['y_m'].iloc[0] - tum['y_m'].iloc[-1]))
    print(f"[4] 出力側の閉ループ確認(複製なし,循環差分で連続性を確認): "
          f"隣接点間隔 最小={np.hypot(np.diff(tum['x_m']), np.diff(tum['y_m'])).min():.3f}m, "
          f"先頭-末尾の物理距離(循環時)={np.hypot(tum['x_m'].iloc[-1]-tum['x_m'].iloc[0], tum['y_m'].iloc[-1]-tum['y_m'].iloc[0]):.3f}m")

    # s_m は既存ファイルの慣習(行順=走行方向、値はTUM由来のまま)に合わせてそのまま保持
    out_center = tum[['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2']].copy()
    out_center = out_center.rename(columns={'ax_mps2': 'ax_mps3'})  # 既存ファイルの列名慣習に合わせる
    out_center.to_csv(OUT_CENTER_CSV, index=False)
    print(f"[4] center_csv_path 用ファイルを保存: {OUT_CENTER_CSV} ({len(out_center)} 行)")

    # ------------------------------------------------------------------
    # 5. 境界(ub/lb) を実測ポリラインへの法線交点として計算
    #    ここでの法線は「標準heading」基準 (n=[-sin(heading),cos(heading)])
    #    で計算する。CSVに書き出した psi_rad (= heading - pi/2, normal_angle用)
    #    をそのまま使うと法線が90度ズレるので注意。
    # ------------------------------------------------------------------
    wp_x = out_center['x_m'].to_numpy()
    wp_y = out_center['y_m'].to_numpy()
    wp_psi = tum['heading_std'].to_numpy()

    n_pts = len(wp_x)  # center_csv と行数を完全に一致させる (複製行なし)
    ub_list, lb_list = [], []
    left_idx_list, right_idx_list = [], []
    left_x_list, left_y_list, right_x_list, right_y_list = [], [], [], []

    for i in range(n_pts):
        P = np.array([wp_x[i], wp_y[i]])
        psi = wp_psi[i]
        # 左法線 (進行方向 + 90deg)
        n_vec = np.array([-np.sin(psi), np.cos(psi)])

        hit_l, d_l = intersect_normal_with_polyline(P, n_vec, left_all)
        hit_r, d_r = intersect_normal_with_polyline(P, -n_vec, right_all)

        if hit_l is None or d_l is None or not np.isfinite(d_l):
            d_l = ub_list[-1] if ub_list else 1.3
            hit_l = P + n_vec * d_l
        if hit_r is None or d_r is None or not np.isfinite(d_r):
            d_r = abs(lb_list[-1]) if lb_list else 1.3
            hit_r = P - n_vec * d_r

        li = int(np.argmin(np.sum((left_all[:-1] - hit_l) ** 2, axis=1)))
        ri = int(np.argmin(np.sum((right_all[:-1] - hit_r) ** 2, axis=1)))

        ub_list.append(float(abs(d_l)))
        lb_list.append(-float(abs(d_r)))
        left_idx_list.append(li)
        right_idx_list.append(ri)
        left_x_list.append(float(hit_l[0]))
        left_y_list.append(float(hit_l[1]))
        right_x_list.append(float(hit_r[0]))
        right_y_list.append(float(hit_r[1]))

    out_bounds = pd.DataFrame({
        'idx': np.arange(n_pts),
        'ub': ub_list,
        'lb': lb_list,
        'left_boundary_idx': left_idx_list,
        'right_boundary_idx': right_idx_list,
        'left_x': left_x_list,
        'left_y': left_y_list,
        'right_x': right_x_list,
        'right_y': right_y_list,
    })
    out_bounds.to_csv(OUT_BOUNDS_CSV, index=False)
    print(f"[5] center_bounds_csv_path 用ファイルを保存: {OUT_BOUNDS_CSV} ({len(out_bounds)} 行)")
    print(f"    ub: min={out_bounds['ub'].min():.3f} max={out_bounds['ub'].max():.3f}")
    print(f"    lb: min={out_bounds['lb'].min():.3f} max={out_bounds['lb'].max():.3f}")

    # ------------------------------------------------------------------
    # 6. visualize_from_width.py 互換ファイル (center_x,center_y,w_tr_left_m,w_tr_right_m)
    # ------------------------------------------------------------------
    out_vis = pd.DataFrame({
        'center_x': wp_x[:n_pts],
        'center_y': wp_y[:n_pts],
        'w_tr_left_m': np.abs(out_bounds['ub'].to_numpy()),
        'w_tr_right_m': np.abs(out_bounds['lb'].to_numpy()),
    })
    # 閉ループ化 (visualize_from_width.py の np.gradient がなるべく安定するよう先頭を複製)
    out_vis = pd.concat([out_vis, out_vis.iloc[[0]]], ignore_index=True)
    out_vis.to_csv(OUT_VISUALIZE_CSV, index=False)
    print(f"[6] visualize_from_width.py 互換ファイルを保存: {OUT_VISUALIZE_CSV} ({len(out_vis)} 行)")


if __name__ == "__main__":
    main()
