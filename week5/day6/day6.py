import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/portscan_clean.csv")
df = df.sample(40000, random_state=42)  # sample keeps the laptop happy
print(df["Label"].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1) Flow duration (log scale, because values range from microseconds to seconds)
df["log_duration"] = np.log10(df["Flow Duration"] + 1)
sns.histplot(data=df, x="log_duration", hue="Label", bins=50, ax=axes[0, 0])
axes[0, 0].set_title("Flow duration (log10 microseconds)")

# 2) Packets per flow (clipped so a few huge flows don't squash the chart)
sns.histplot(
    data=df[df["Total Fwd Packets"] <= 20],
    x="Total Fwd Packets",
    hue="Label",
    bins=20,
    ax=axes[0, 1],
)
axes[0, 1].set_title("Forward packets per flow (up to 20)")

# 3) Average packet size
sns.boxplot(data=df, x="Label", y="Average Packet Size", ax=axes[1, 0])
axes[1, 0].set_title("Average packet size")

# 4) Top destination ports per class
top = df.groupby("Label")["Destination Port"].value_counts().groupby(level=0).head(6)
top = top.reset_index(name="count")
top["Destination Port"] = top["Destination Port"].astype(str)
sns.barplot(data=top, x="Destination Port", y="count", hue="Label", ax=axes[1, 1])
axes[1, 1].set_title("Most common destination ports")

plt.tight_layout()
plt.savefig("week5/day6/attack_vs_normal.png", dpi=120)
print("Saved week5/day6/attack_vs_normal.png")
