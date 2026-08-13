import pandas as pd
import numpy as np

# ======================
# 読み込み
# ======================
df = pd.read_csv(
    "traj_race_cl.csv",
    sep=";",
    comment="#",
    names=[
        "s_m",
        "x_m",
        "y_m",
        "psi_rad",
        "kappa_radpm",
        "vx_mps",
        "ax_mps2"
    ]
)

# ======================
# reverse
# ======================
df = df.iloc[::-1].reset_index(drop=True)



# ======================
# 座標補正
# ======================
X_OFFSET = 5.332886
Y_OFFSET = -75.727413

df["x_m"] += X_OFFSET
df["y_m"] += Y_OFFSET


# ======================
# ★ここが重要：重複点の削除
# ======================
x = df["x_m"].to_numpy()
y = df["y_m"].to_numpy()

dist_to_next = np.sqrt(
    (np.roll(x, -1) - x)**2 +
    (np.roll(y, -1) - y)**2
)

# 最後と最初が同一点 or ほぼ同一点なら削除
# （閉ループ重複対策）
if dist_to_next[-1] < 1e-3:
    df = df.iloc[:-1].reset_index(drop=True)

# 再取得（重要）
x = df["x_m"].to_numpy()
y = df["y_m"].to_numpy()

# ======================
# 閉ループ差分
# ======================
dx = np.roll(x, -1) - x
dy = np.roll(y, -1) - y

ds_closed = np.sqrt(dx**2 + dy**2)

# ======================
# s再計算
# ======================
s = np.concatenate(([0.0], np.cumsum(ds_closed[:-1])))
df["s_m"] = s

# ======================
# yaw再計算
# ======================
psi = np.arctan2(dy, dx)

# unwrapして連続化
psi_unwrap = np.unwrap(psi)

# 出力用は[-π,π]
df["psi_rad"] = (psi_unwrap + np.pi) % (2*np.pi) - np.pi

# ======================
# 曲率再計算
# ======================
dpsi = np.roll(psi_unwrap, -1) - psi_unwrap

kappa = dpsi / np.maximum(ds_closed, 1e-6)

# 異常値除去（任意）
kappa = np.clip(kappa, -1.0, 1.0)

df["kappa_radpm"] = kappa

# ======================
# MPC用 csv
# ======================
mpc_df = pd.DataFrame({
    "s_m": df["s_m"],
    "x_m": df["x_m"],
    "y_m": df["y_m"],
    "psi_rad": df["psi_rad"],
    "kappa_radpm": df["kappa_radpm"],
    "vx_mps": df["vx_mps"],
    "ax_mps3": df["ax_mps2"]
})

mpc_df.to_csv(
    "traj_race_cl_mpc.csv",
    index=False,
    float_format="%.10f"
)

print("saved traj_race_cl_mpc.csv")

# ======================
# simple_trajectory_generator用
# ======================
yaw = df["psi_rad"].to_numpy()

qz = np.sin(yaw / 2.0)
qw = np.cos(yaw / 2.0)

editor_df = pd.DataFrame({
    "x": df["x_m"],
    "y": df["y_m"],
    "z": np.zeros(len(df)),
    "x_quat": np.zeros(len(df)),
    "y_quat": np.zeros(len(df)),
    "z_quat": qz,
    "w_quat": qw,
    "speed": df["vx_mps"]
})

editor_df.to_csv(
    "traj_race_cl_editor.csv",
    index=False,
    float_format="%.10f"
)

print("saved traj_race_cl_editor.csv")