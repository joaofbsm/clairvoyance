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

df = pd.Series({"ChampHistWinRateDiff": 175, "ChampHistBlueWinRate": 43, "ChampHistWinTotalDiff": 36, "ChampHistRedWinRate":35, "ChampHistBlueWinTotal": 28, "ChampMasteryBlueSumm1": 22, "ChampMasteryBlueSumm4": 20, "ChampMasteryRedSumm4": 20, "ChampHistRedWinTotal": 19, "ChampMasteryBlueSumm2": 18}, name="feat_import")

matplotlib.rc('xtick', labelsize=5) 

#print(df)
#print(df.sort_values(ascending=False))

df = df.sort_values(ascending=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
ax.xaxis.set_tick_params(labelsize=5)
df.plot(ax=ax, kind='bar', use_index=True)

#ax.yaxis.set_ticklabels(df.League)
#ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_facecolor("white")
ax.grid(axis="x")
#plt.title(u"Distribuição de Jogadores por tier", fontsize=12)
#plt.show()
plt.savefig("feat_plot.png", dpi=350, transparent=True, bbox_inches="tight")