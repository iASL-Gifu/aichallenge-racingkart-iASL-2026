import pandas as pd
import numpy as np

df = pd.read_csv("traj_race_cl_fixed.csv")

# 全行を逆順にする
df = df.iloc[::-1].reset_index(drop=True)

x = df["x_m"].values
y = df["y_m"].values

# s再計算
ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
s = np.concatenate(([0.0], np.cumsum(ds)))

# ψ再計算
psi = np.arctan2(np.diff(y), np.diff(x))
psi = np.append(psi, psi[-1])

# κ計算用に連続化
psi_unwrap = np.unwrap(psi)

# κ再計算
kappa = np.gradient(psi_unwrap) / np.gradient(s)

# ψを[-π, π]に戻す
psi = (psi_unwrap + np.pi) % (2*np.pi) - np.pi

df["s_m"] = s
df["psi_rad"] = psi
df["kappa_radpm"] = kappa

df.to_csv("traj_race_cl_reversed.csv", index=False)