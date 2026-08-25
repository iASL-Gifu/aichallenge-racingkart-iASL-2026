import pandas as pd
import numpy as np

df = pd.read_csv("centerline_devided.csv")

# 元の値を保存
old_left = df["w_tr_left_m"].copy()
old_right = df["w_tr_right_m"].copy()

# ユークリッド距離で再計算
df["w_tr_left_m"] = np.sqrt(
    (df["left_x"] - df["center_x"])**2 +
    (df["left_y"] - df["center_y"])**2
)

df["w_tr_right_m"] = np.sqrt(
    (df["right_x"] - df["center_x"])**2 +
    (df["right_y"] - df["center_y"])**2
)

df["width"] = df["w_tr_left_m"] + df["w_tr_right_m"]

# 変化した行を判定（浮動小数点誤差を考慮）
changed = (
    ~np.isclose(old_left, df["w_tr_left_m"], atol=1e-6) |
    ~np.isclose(old_right, df["w_tr_right_m"], atol=1e-6)
)

print(f"変更された行数: {changed.sum()} / {len(df)}")

# 変更された行だけ表示したい場合
print(df.loc[changed, [
    "w_tr_left_m",
    "w_tr_right_m"
]])

df.to_csv("centerline_devided_recomputed.csv", index=False)