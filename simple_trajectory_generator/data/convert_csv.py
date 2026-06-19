import pandas as pd
import numpy as np

# 読み込み元と出力先のファイルパス
input_csv = "traj_min.csv"
output_csv = "traj_mincurv_converted.csv"

# データの読み込み
df = pd.read_csv(input_csv)

# ヨー角 (psi_rad) からクォータニオンを計算
# yaw から qz, qw への変換 (qx=0, qy=0)
yaw = df['psi_rad']
qz = np.sin(yaw / 2.0)
qw = np.cos(yaw / 2.0)

# 新しいデータフレームを作成 (8列)
new_df = pd.DataFrame({
    'x': df['x_m'],
    'y': df['y_m'],
    'z': 0.0,
    'x_quat': 0.0,
    'y_quat': 0.0,
    'z_quat': qz,
    'w_quat': qw,
    'speed': df['vx_mps']
})

# CSVとして保存
new_df.to_csv(output_csv, index=False)
print(f"変換が完了しました: {output_csv}")
