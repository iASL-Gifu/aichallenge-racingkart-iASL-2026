#!/usr/bin/env python3
"""
raceline_converted_Xm.csv を MPC が読める traj_mincurv.csv 互換形式に変換するスクリプト

入力フォーマット: x, y, z, x_quat, y_quat, z_quat, w_quat, speed
出力フォーマット: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2

使い方:
  python3 convert_raceline.py raceline_converted_2m.csv output.csv
  python3 convert_raceline.py raceline_converted_2m.csv output.csv --speed_scale 0.8  # 全体速度を80%に
  python3 convert_raceline.py raceline_converted_2m.csv output.csv --max_speed 20.0   # 最高速度を20km/hに制限
"""

import sys
import argparse
import numpy as np
import pandas as pd


def quaternion_to_yaw(z_quat, w_quat):
    """z-w クォータニオン（x=y=0）から yaw 角を計算"""
    return 2.0 * np.arctan2(z_quat, w_quat)


def compute_curvature(x, y):
    """
    離散点列から曲率 kappa [rad/m] を計算する
    中心差分を使用（端点は前進/後退差分）
    """
    n = len(x)
    kappa = np.zeros(n)

    for i in range(n):
        # 周回コースなのでインデックスをラップ
        i_prev = (i - 1) % n
        i_next = (i + 1) % n

        dx1 = x[i]      - x[i_prev]
        dy1 = y[i]      - y[i_prev]
        dx2 = x[i_next] - x[i]
        dy2 = y[i_next] - y[i]

        # クロス積（符号付き面積）と距離
        cross = dx1 * dy2 - dy1 * dx2
        d1 = np.hypot(dx1, dy1)
        d2 = np.hypot(dx2, dy2)

        denom = d1 * d2 * np.hypot(dx1 + dx2, dy1 + dy2)
        if denom < 1e-10:
            kappa[i] = 0.0
        else:
            kappa[i] = 2.0 * cross / denom

    return kappa


def compute_arc_length(x, y):
    """累積弧長 s [m] を計算"""
    s = [0.0]
    for i in range(1, len(x)):
        ds = np.hypot(x[i] - x[i-1], y[i] - y[i-1])
        s.append(s[-1] + ds)
    return np.array(s)


def compute_acceleration(vx, s):
    """速度と弧長から加速度 ax [m/s^2] を計算（v*dv/ds）"""
    n = len(vx)
    ax = np.zeros(n)
    for i in range(n):
        i_next = (i + 1) % n
        ds = s[i_next] - s[i] if i < n - 1 else s[1] - s[0]
        if abs(ds) < 1e-10:
            ax[i] = 0.0
        else:
            ax[i] = (vx[i_next] - vx[i]) / ds * vx[i]  # a = v * dv/ds
    return ax


def convert(input_csv, output_csv, speed_scale=1.0, max_speed=None):
    # 入力読み込み
    df = pd.read_csv(input_csv)
    print(f"入力: {input_csv}  ({len(df)} 点)")
    print(f"列名: {list(df.columns)}")

    x = df['x'].values
    y = df['y'].values
    z_quat = df['z_quat'].values
    w_quat = df['w_quat'].values
    speed_kmh = df['speed'].values  # 単位は km/h

    # psi_rad: クォータニオンから yaw を計算
    psi = quaternion_to_yaw(z_quat, w_quat)

    # 弧長
    s = compute_arc_length(x, y)

    # 曲率
    kappa = compute_curvature(x, y)

    # 速度変換: km/h → m/s、スケーリングと上限適用
    vx = speed_kmh / 3.6 * speed_scale
    if max_speed is not None:
        vx = np.minimum(vx, max_speed / 3.6)

    # 加速度
    ax = compute_acceleration(vx, s)

    # 出力 DataFrame 作成
    out_df = pd.DataFrame({
        's_m':          np.round(s, 7),
        'x_m':          np.round(x, 7),
        'y_m':          np.round(y, 7),
        'psi_rad':      np.round(psi, 7),
        'kappa_radpm':  np.round(kappa, 7),
        'vx_mps':       np.round(vx, 7),
        'ax_mps2':      np.round(ax, 7),
    })

    out_df.to_csv(output_csv, index=False)
    print(f"出力: {output_csv}  ({len(out_df)} 点)")
    print(f"  速度範囲: {vx.min()*3.6:.1f} ~ {vx.max()*3.6:.1f} km/h")
    print(f"  弧長合計: {s[-1]:.1f} m")
    print("完了!")


def main():
    parser = argparse.ArgumentParser(
        description="raceline_converted_Xm.csv → traj_mincurv.csv 互換形式に変換")
    parser.add_argument("input",  help="入力CSVパス (x,y,z,x_quat,...,speed形式)")
    parser.add_argument("output", help="出力CSVパス")
    parser.add_argument("--speed_scale", type=float, default=1.0,
                        help="速度スケール係数 (例: 0.8 で80%%に減速, デフォルト: 1.0)")
    parser.add_argument("--max_speed", type=float, default=None,
                        help="最高速度上限 [km/h] (例: 20.0, デフォルト: 制限なし)")
    args = parser.parse_args()

    convert(args.input, args.output, args.speed_scale, args.max_speed)


if __name__ == "__main__":
    main()
