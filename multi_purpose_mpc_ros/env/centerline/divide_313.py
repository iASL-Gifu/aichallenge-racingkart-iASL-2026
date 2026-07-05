import pandas as pd
import numpy as np

# ==========================
# 設定
# ==========================
input_csv = "boundary.csv"
output_csv = "centerline_devided.csv"

NUM_POINTS = 313
#=========================
# CSV読み込み
#=========================
df = pd.read_csv(input_csv)

#=========================
# 累積距離
#=========================
x = df["center_x"].values
y = df["center_y"].values

dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
s = np.insert(np.cumsum(dist), 0, 0.0)

#=========================
# 等間隔な距離
#=========================
target_s = np.linspace(0, s[-1], NUM_POINTS)

#=========================
# 全列を補間
#=========================
new_df = pd.DataFrame()

for col in df.columns:
    new_df[col] = np.interp(target_s, s, df[col].values)

#=========================
# 保存
#=========================
new_df.to_csv(output_csv, index=False)

print(f"元データ : {len(df)} 点")
print(f"補間後 : {len(new_df)} 点")
print(f"コース長 : {s[-1]:.3f} m")
