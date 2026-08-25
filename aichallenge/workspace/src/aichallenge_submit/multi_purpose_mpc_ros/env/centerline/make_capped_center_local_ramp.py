"""center_converted.csv (実測 w_tr_right/left) から、mincurv_iqp 入力用の
center_converted_capped.csv を作る。ただし特定区間だけ CAP を引き上げ、
前後はコサインカーブでランプさせて滑らかに繋ぐ。

背景:
  従来の center_converted_capped.csv は全周一律 CAP=1.3m でクリップされている。
  以下の CORE_REGIONS で指定した区間は実測コース幅に余裕があるのに 1.3m に
  制限されていたため、mincurv_iqp がこの余裕を使えず、不要にきついS字
  (切り返し)を生成していた。該当区間だけ CAP を引き上げて、オプティマイザに
  センターラインを素直な(低曲率な)形へ均してもらう。

  region 1: idx513-629 (traj_center_mincurv_capped.csv の idx245-265,
            s_m≈92-112m) 元々のS字/シケイン。TARGET_CAP=2.5m。
  region 2: idx1140-1287 (traj_center_mincurv_capped.csv の idx134-149,
            s_m≈200-219m) 2つ目の高曲率反転区間。最狭部が右2.50m/左2.41m
            しかないため TARGET_CAP=2.0m に抑える。

使い方:
  python3 make_capped_center_local_ramp.py
  出力: center_converted_capped_local_ramp.csv
  (既存の center_converted_capped.csv は上書きしない。
   問題なければ手動でリネーム/差し替えてから mincurv_iqp を実行してください)
"""
import os
import numpy as np
import pandas as pd

TRACKS_DIR = "/home/takenoyama/Documents/AI_challenge/global_racetrajectory_optimization/inputs/tracks"
IN_CSV = os.path.join(TRACKS_DIR, "center_converted.csv")
OUT_CSV = os.path.join(TRACKS_DIR, "center_converted_capped_local_ramp.csv")

# 全周デフォルトのCAP (従来通り)
BASE_CAP = 1.3

# 安全マージン: 実測幅ぎりぎりまでは使わせない (壁からのクリアランス確保)
SAFETY_MARGIN = 0.3

# CAPを引き上げる区間のリスト。各要素は
#   lo, hi          : center_converted.csv の行インデックス (0-origin, 閉区間, 核区間)
#   ramp_points_lo  : lo側 (i<lo, 走行方向でいう「核区間の後」側) のランプ長 [点数]
#   ramp_points_hi  : hi側 (i>hi, 走行方向でいう「核区間の手前」側) のランプ長 [点数]
#   target_cap      : 核区間で目指すCAP [m]
# 走行方向: このCSVでは走行が進むにつれて center_converted.csv 側のインデックスが
# 減少する向きになっている (traj_center_mincurv_capped.csv の waypoint 増加 =
# center_converted.csv の idx 減少、を実測で確認済み)。そのため「核区間の直後
# しばらくカーブが続く」場合は lo 側 (i<lo) のランプを伸ばす。
CORE_REGIONS = [
    # 元のS字/シケイン (traj idx245-265, s_m≈92-112m)。実測最小幅(右3.02m/左3.33m)。
    dict(lo=513, hi=629, ramp_points_lo=200, ramp_points_hi=200, target_cap=2.1),
    # 2つ目の高曲率反転区間 (traj idx134-149, s_m≈200-219m)。
    # 実測最小幅が右2.50m/左2.41mとやや狭いため TARGET_CAP は抑えめ。
    # 区間直後 (走行方向で lo 側 = idx1140未満) もしばらくカーブが続くため、
    # そちら側のランプだけ手前側(hi側)の2倍にしている。
    dict(lo=1140, hi=1287, ramp_points_lo=350, ramp_points_hi=100, target_cap=1.6),
]


def region_weight(i: int, n: int, lo: int, hi: int, ramp_points_lo: int, ramp_points_hi: int) -> float:
    """行インデックス i (0-origin, 周回mod n) における、1つの core 区間に対する
    昇圧の重み [0,1] を返す。core区間内は1.0、その前後は非対称なランプ長で
    コサインで0→1→0に遷移、それ以外は0。周回(先頭/末尾接続)を考慮する。
    """
    def circ_dist(a, b):
        d = abs(a - b) % n
        return min(d, n - d)

    if lo <= i <= hi:
        return 1.0

    if i < lo:
        d = lo - i
        ramp_points = ramp_points_lo
    else:
        d = i - hi
        ramp_points = ramp_points_hi

    d = min(d, circ_dist(i, lo), circ_dist(i, hi))

    if d > ramp_points:
        return 0.0
    return 0.5 * (1.0 + np.cos(np.pi * d / ramp_points))


def main():
    tc = pd.read_csv(IN_CSV, comment='#', header=None,
                      names=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
    n = len(tc)

    real_right = tc['w_tr_right_m'].to_numpy()
    real_left = tc['w_tr_left_m'].to_numpy()

    # 各区間ごとの ramped_cap を求め、点ごとに最大値を採用する
    # (区間が重ならない前提だが、念のためmaxで合成しておく)
    ramped_cap = np.full(n, BASE_CAP)
    any_weight = np.zeros(n, dtype=bool)

    for region in CORE_REGIONS:
        weights = np.array([region_weight(i, n, region['lo'], region['hi'],
                                           region['ramp_points_lo'], region['ramp_points_hi'])
                             for i in range(n)])
        region_cap = BASE_CAP + weights * (region['target_cap'] - BASE_CAP)
        mask = weights > 0
        ramped_cap = np.where(mask, np.maximum(ramped_cap, region_cap), ramped_cap)
        any_weight |= mask

    out_right = np.minimum(ramped_cap, real_right - SAFETY_MARGIN)
    out_left = np.minimum(ramped_cap, real_left - SAFETY_MARGIN)
    # 実測幅がSAFETY_MARGIN未満で負になるケースの保険 (通常は発生しない想定)
    out_right = np.maximum(out_right, 0.3)
    out_left = np.maximum(out_left, 0.3)
    # 核・ランプ区間外は従来通りBASE_CAP (実測がBASE_CAP未満ならそちらを優先)
    out_right = np.where(any_weight, out_right, np.minimum(BASE_CAP, real_right))
    out_left = np.where(any_weight, out_left, np.minimum(BASE_CAP, real_left))

    out = pd.DataFrame({
        'x_m': tc['x_m'],
        'y_m': tc['y_m'],
        'w_tr_right_m': out_right,
        'w_tr_left_m': out_left,
    })

    header = "# x_m,y_m,w_tr_right_m,w_tr_left_m"
    with open(OUT_CSV, 'w') as f:
        f.write(header + "\n")
        out.to_csv(f, index=False, header=False)

    print(f"[done] {OUT_CSV} ({n} 行)")
    print(f"CAP引き上げ対象: {any_weight.sum()} 点 ({len(CORE_REGIONS)} 区間)")
    for region in CORE_REGIONS:
        lo, hi = region['lo'], region['hi']
        print(f"  核 idx{lo}-{hi}, "
              f"ランプ lo側={region['ramp_points_lo']}点/"
              f"hi側={region['ramp_points_hi']}点, "
              f"目標TARGET_CAP={region['target_cap']}m: "
              f"CAP最大値 right={out_right[lo:hi+1].max():.3f}m left={out_left[lo:hi+1].max():.3f}m")


if __name__ == "__main__":
    main()
