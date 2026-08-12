import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# CSV読み込み
# ===========================
csv_file = "centerline_devided.csv"   # ←ファイル名を変更してください

df = pd.read_csv(csv_file)
print("centerline_devided.csv")

# ===========================
# 可視化
# ===========================
plt.figure(figsize=(10, 10))

# 左境界
plt.plot(
    df["left_x"],
    df["left_y"],
    color="red",
    linewidth=2,
    label="Left Boundary"
)

# 右境界
plt.plot(
    df["right_x"],
    df["right_y"],
    color="blue",
    linewidth=2,
    label="Right Boundary"
)

# センターライン
plt.plot(
    df["center_x"],
    df["center_y"],
    color="black",
    linewidth=1.5,
    linestyle="--",
    label="Center Line"
)

# 数点ごとに左右を結ぶ線を描画
step = 20  # 間隔（小さくすると見やすいが重くなる）
for i in range(0, len(df), step):
    plt.plot(
        [df["left_x"][i], df["right_x"][i]],
        [df["left_y"][i], df["right_y"][i]],
        color="gray",
        alpha=0.5,
        linewidth=0.8
    )

plt.axis("equal")
plt.grid(True)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Track Boundaries and Center Line")
plt.legend()

plt.tight_layout()
plt.show()
