#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xgboost
import global_constants as gc
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

dataset_name = sys.argv[1]

# load data
dataset = loadtxt(dataset_name + ".csv", delimiter=",")
input_size = dataset.shape[1] - 1

# split data into X and y
X = dataset[:,0:input_size]
y = dataset[:,input_size]

# fit model no training data
#model = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
model = RandomForestClassifier(n_estimators=100, max_depth=2)

kfold = KFold(n_splits=5, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
