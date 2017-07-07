#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

from matplotlib import rcParams
rcParams['axes.titlepad'] = 20

reload(sys)
sys.setdefaultencoding("latin-1")

matplotlib.rc('font', family='DejaVu Sans')
plt.style.use("bmh")

ax = plt.subplot()

df = pd.read_csv("player_leagues.csv", names=["League", "Players"])
#print(df)
order = ["SEM RANKING", "BRONZE", "PRATA", "OURO", "PLATINA", "DIAMANTE", "MESTRE", "DESAFIANTE"]
mapping = {league: i for i, league in enumerate(order)}
key = df["League"].map(mapping)
df = df.iloc[key.argsort()]
#print(df)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)

df.Players.plot(ax=ax, kind='pie', colors=["#808080", "#CD7F32", "#C0C0C0", "#FFD700", "#E5E4E2", "#0EBFE9", "#800080", "#A60628"], legend=True, use_index=True)

ax.yaxis.set_ticklabels(df.League)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_facecolor("white")
ax.grid(axis="y")
#plt.title(u"Distribuição de Jogadores por tier", fontsize=12)
plt.show()
plt.savefig("league_plot.png", dpi=350, transparent=True, bbox_inches="tight")