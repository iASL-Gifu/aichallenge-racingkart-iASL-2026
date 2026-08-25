#!/usr/bin/env python3
"""624点CSVから1点おきに抽出して312点CSVを生成する。

geometry（kappa, psi, s_m）は抽出後の座標から再計算して整合性を保つ。
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
MPC_CURVATURE_WINDOW = 7  # 624点版スクリプトと同じ


def mpc_discrete_geometry(x: np.ndarray, y: np.ndarray, curvature_window: int):
    """ReferencePath と同じ離散ヘディング・曲率計算。"""
    dx_ahead = np.roll(x, -1) - x
    dy_ahead = np.roll(y, -1) - y
    segment = np.hypot(dx_ahead, dy_ahead)
    heading = np.arctan2(dy_ahead, dx_ahead)
    heading_behind = np.arctan2(y - np.roll(y, 1), x - np.roll(x, 1))
    heading_change = np.angle(np.exp(1j * (heading - heading_behind)))
    kappa = heading_change / segment
    kappa = savgol_filter(kappa, curvature_window, 3, mode="wrap")
    return heading, kappa


def main() -> None:
    # ── 元CSVの読み込み ──
    traj624 = pd.read_csv(HERE / "traj_center624_local_smooth.csv")
    bounds624 = pd.read_csv(HERE / "waypoint_bounds_center624_local_smooth.csv")

    # ── 1点おきに抽出（インデックス 0,2,4,...,622 → 312点） ──
    traj312 = traj624.iloc[::2].reset_index(drop=True)
    bounds312 = bounds624.iloc[::2].reset_index(drop=True)

    x = traj312["x_m"].to_numpy()
    y = traj312["y_m"].to_numpy()

    # ── geometry の再計算 ──
    heading, kappa = mpc_discrete_geometry(x, y, MPC_CURVATURE_WINDOW)
    actual_segment = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    s = np.r_[0.0, np.cumsum(actual_segment[:-1])]

    # psi_rad は boundary_normal_angle = heading - pi/2（624版と同じ規則）
    boundary_normal_angle = np.unwrap(heading - np.pi / 2.0)

    # ── trajectory CSV の構築 ──
    traj_out = pd.DataFrame({
        "s_m":          s,
        "x_m":          x,
        "y_m":          y,
        "psi_rad":      boundary_normal_angle,
        "kappa_radpm":  kappa,
        "vx_mps":       traj312["vx_mps"].to_numpy(),
        "ax_mps3":      traj312["ax_mps3"].to_numpy(),
    })

    # ── bounds CSV のインデックス更新 ──
    bounds_out = bounds312.copy()
    bounds_out["idx"] = np.arange(len(bounds_out))

    # ── 保存 ──
    out_traj   = HERE / "traj_center312_local_smooth.csv"
    out_bounds = HERE / "waypoint_bounds_center312_local_smooth.csv"
    traj_out.to_csv(out_traj, index=False, float_format="%.10f")
    bounds_out.to_csv(out_bounds, index=False, float_format="%.10f")

    # ── 統計の表示 ──
    total_length = float(actual_segment.sum())
    print(f"生成完了: {len(traj_out)} 点 / 全長 {total_length:.3f} m")
    print(f"  kappa  : {kappa.min():.6f} .. {kappa.max():.6f} rad/m")
    print(f"  bounds ub min={bounds_out.ub.min():.3f} m  lb min={(-bounds_out.lb).min():.3f} m")
    print(f"  出力: {out_traj}")
    print(f"  出力: {out_bounds}")


if __name__ == "__main__":
    main()
