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


df = pd.Series([20397, 5885, 5742, 2653, 2006, 994, 494, 96],
                index=['DIAMANTE', 'PLATINA', 'SEM RANKING', 'MESTRE', 'OURO', 'DESAFIANTE', 'PRATA', 'BRONZE'])

order = ["SEM RANKING", "BRONZE", "PRATA", "OURO", "PLATINA", "DIAMANTE", "MESTRE", "DESAFIANTE"]

colors = ["#0EBFE9", "#E5E4E2", "#808080", "#800080", "#FFD700", "#A60628", "#C0C0C0", "#CD7F32"]

explode = (0, 0, 0, 0.1, 0, 0, 0, 0)

#df.plot(kind='pie', fontsize=12, colors=colors, explode=explode)
df.plot(kind='bar', fontsize=12, color=colors)
plt.axis('equal')
plt.ylabel('')
plt.legend(labels=df.index, loc="best")

plt.show()