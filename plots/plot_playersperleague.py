import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = "DejaVu Sans"
plt.style.use("bmh")

ax = plt.subplot()

df = pd.read_csv("player_leagues.csv", names=["League", "Players"])

order = ["UNRANKED", "BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND", "MASTER", "CHALLENGER"]
mapping = {league: i for i, league in enumerate(order)}
key = df["League"].map(mapping)
df = df.iloc[key.argsort()]

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

df.Players.plot(ax=ax, kind='barh', color=["#808080", "#CD7F32", "#C0C0C0", "#FFD700", "#E5E4E2", "#0EBFE9", "#800080", "#A60628"], legend=False, use_index=False)

ax.yaxis.set_ticklabels(df.League)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_facecolor("white")
ax.grid(axis="y")
plt.suptitle("Jogadores por Liga", fontsize=14)
plt.show()