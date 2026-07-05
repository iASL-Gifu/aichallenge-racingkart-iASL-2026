import pandas as pd
import numpy as np

# ==========================
# 設定
# ==========================
INPUT_CSV = "../env/min_curv/centerline_mpc_1m.csv"
OUTPUT_CSV1 = "../env/min_curv/traj_center.csv"
OUTPUT_CSV2 = "../env/waypoint_bounds_center.csv"

VX = 9.694
AX = 0.0

# ==========================
# 読み込み
# ==========================
df = pd.read_csv(INPUT_CSV)

print(df.columns.tolist())

# 入力CSV
#left_x,left_y,right_x,right_y,center_x,center_y,width,w_tr_left_m,w_tr_right_m
x = df["center_x"].values
y = df["center_y"].values
w_right = df["w_tr_right_m"].values
w_left = df["w_tr_left_m"].values

# ==========================
# yaw・曲率計算
# ==========================
dx = np.gradient(x)
dy = np.gradient(y)

ds = np.sqrt(dx**2 + dy**2)

# 0除算防止
ds[ds < 1e-8] = 1e-8

psi = np.unwrap(np.arctan2(dy, dx))

dpsi = np.gradient(psi)
kappa = dpsi / ds

# ==========================
# s_m
# ==========================
N = len(df)
s_m = np.arange(N - 1, -1, -1)

# ==========================
# trajectory.csv 作成
# ==========================
traj = pd.DataFrame({
    "s_m": s_m,
    "x_m": x,
    "y_m": y,
    "psi_rad": psi,
    "kappa_radpm": kappa,
    "vx_mps": np.full(N, VX),
    "ax_mps3": np.full(N, AX)
})

traj.to_csv(
    OUTPUT_CSV1,
    index=False,
    float_format="%.10f"
)

print(f"Saved : {OUTPUT_CSV1}")

# ==========================
# waypoint_bounds.csv 作成
# ==========================
waypoint_bounds = pd.DataFrame({
    "idx": np.arange(N),
    "ub": w_left,
    "lb": -w_right,
    "distance": np.zeros(N)
})

waypoint_bounds.to_csv(
    OUTPUT_CSV2,
    index=False,
    float_format="%.15f"
)

print(f"Saved : {OUTPUT_CSV2}")