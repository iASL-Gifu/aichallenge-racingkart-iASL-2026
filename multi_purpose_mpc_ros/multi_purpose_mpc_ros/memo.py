import pandas as pd
import numpy as np

df = pd.read_csv("../../global_racetrajectory_optimization/outputs/traj_race_cl_mpc.csv")

print("points =", len(df))

x = df["x_m"].to_numpy()
y = df["y_m"].to_numpy()

# ------------------
# 幾何
# ------------------
dx = np.diff(x)
dy = np.diff(y)

ds_geom = np.hypot(dx, dy)

print("\n===== Geometry ds =====")
print("min =", ds_geom.min())
print("max =", ds_geom.max())
print("mean =", ds_geom.mean())

# ------------------
# s増分
# ------------------
s = df["s_m"].to_numpy()

ds_s = np.diff(s)

print("\n===== s increment =====")
print("min =", ds_s.min())
print("max =", ds_s.max())
print("mean =", ds_s.mean())

# sが単調減少か？
print("\n===== s direction =====")
print("all positive =", np.all(ds_s > 0))
print("all negative =", np.all(ds_s < 0))

# ------------------
# psi
# ------------------
psi = np.unwrap(df["psi_rad"].to_numpy())

dpsi = np.diff(psi)

print("\n===== psi =====")
print("max abs dpsi =", np.max(np.abs(dpsi)))

# ------------------
# kappa
# ------------------
kappa = df["kappa_radpm"].to_numpy()

print("\n===== kappa =====")
print("min =", kappa.min())
print("max =", kappa.max())

# 符号反転回数
sign_change = np.sum(
    np.sign(kappa[:-1]) != np.sign(kappa[1:])
)

print("sign changes =", sign_change)

# スパイク
idx = np.where(np.abs(kappa) > 0.3)[0]

print("\n===== large kappa =====")
print("count =", len(idx))

for i in idx:
    print(
        f"idx={i} "
        f"s={s[i]:.2f} "
        f"kappa={kappa[i]:.3f}"
    )

# ------------------
# ψと接線方向比較
# ------------------
yaw_geom = np.arctan2(dy, dx)

psi_trim = psi[:-1]

err = np.arctan2(
    np.sin(psi_trim - yaw_geom),
    np.cos(psi_trim - yaw_geom)
)

print("\n===== psi vs geometry =====")
print("mean err deg =", np.degrees(np.mean(np.abs(err))))
print("max err deg  =", np.degrees(np.max(np.abs(err))))