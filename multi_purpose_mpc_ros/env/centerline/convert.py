import pandas as pd

X_OFFSET = 5.332886
Y_OFFSET = -75.727413

# 読み込み
df = pd.read_csv("traj_center313.csv")

# 座標補正
df["x_m"] = df["x_m"] + X_OFFSET
df["y_m"] = df["y_m"] + Y_OFFSET

# 保存
df.to_csv("traj_center313_corrected.csv", index=False)