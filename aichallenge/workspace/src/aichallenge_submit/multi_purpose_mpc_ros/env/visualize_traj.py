import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("env/boundary.csv")

# 描画
plt.figure(figsize=(10, 10))

# 左境界
plt.plot(
    df["left_x"],
    df["left_y"],
    label="Left Boundary",
    linewidth=2
)

# 右境界
plt.plot(
    df["right_x"],
    df["right_y"],
    label="Right Boundary",
    linewidth=2
)

# センターライン
plt.plot(
    df["center_x"],
    df["center_y"],
    'k--',
    label="Centerline",
    linewidth=1.5
)

# 開始点
plt.scatter(
    df["left_x"].iloc[0],
    df["left_y"].iloc[0],
    marker="o",
    s=80,
    label="Left Start"
)

plt.scatter(
    df["right_x"].iloc[0],
    df["right_y"].iloc[0],
    marker="o",
    s=80,
    label="Right Start"
)

# 終了点
plt.scatter(
    df["left_x"].iloc[-1],
    df["left_y"].iloc[-1],
    marker="x",
    s=100,
    label="Left End"
)

plt.scatter(
    df["right_x"].iloc[-1],
    df["right_y"].iloc[-1],
    marker="x",
    s=100,
    label="Right End"
)

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Track Boundaries")
plt.axis("equal")      # 縦横比を実際の距離に合わせる
plt.grid(True)
plt.legend()

plt.show()