import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("traj_race_cl_reversed.csv")

plt.figure(figsize=(8,8))
plt.plot(df["x_m"], df["y_m"], "r")
plt.axis("equal")
plt.show()