import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CSV読み込み
# ======================
df = pd.read_csv("/home/haruki/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/env/final_ver4/traj_mincurv.csv")

x = df["x_m"].to_numpy()
y = df["y_m"].to_numpy()
s = df["s_m"].to_numpy()

# ======================
# 軌跡描画
# ======================
plt.figure(figsize=(10, 10))

plt.plot(
    x,
    y,
    'k-',
    linewidth=1.5,
    label="trajectory"
)

# ======================
# s表示
# 10mごとに表示
# ======================
target_step = 10.0

s_abs = np.abs(s)

targets = np.arange(
    np.min(s_abs),
    np.max(s_abs) + target_step,
    target_step
)

for target in targets:

    idx = np.argmin(np.abs(s_abs - target))

    plt.text(
        x[idx],
        y[idx],
        f"{s[idx]:.0f}",
        fontsize=8
    )

    plt.plot(
        x[idx],
        y[idx],
        'ro',
        markersize=3
    )

# ======================
# 開始点・終了点
# ======================
plt.plot(
    x[0],
    y[0],
    'go',
    markersize=10,
    label="start"
)

plt.plot(
    x[-1],
    y[-1],
    'bo',
    markersize=10,
    label="end"
)

# ======================
# 軸設定
# ======================
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.title("Trajectory with s_m")

plt.tight_layout()
plt.show()