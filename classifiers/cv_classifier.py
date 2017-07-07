#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xgboost
import global_constants as gc
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

dataset_name = sys.argv[1]

# load data
dataset = loadtxt(dataset_name + ".csv", delimiter=",")
input_size = dataset.shape[1] - 1

# split data into X and y
X = dataset[:,0:input_size]
y = dataset[:,input_size]

# fit model to training data
#adamodel = AdaBoostClassifier()
#xgbmodel = xgboost.XGBClassifier()
#rfmodel = RandomForestClassifier()

# best hyperparameters without history
#adamodel = AdaBoostClassifier(n_estimators=50, learning_rate=1)
#xgbmodel = xgboost.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=2)
#rfmodel = RandomForestClassifier(n_estimators=100, max_depth=2)

# best hyperparameters with history
adamodel = AdaBoostClassifier(n_estimators=300, learning_rate=0.1)
xgbmodel = xgboost.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=2)
rfmodel = RandomForestClassifier(n_estimators=100, max_depth=2)

kfold = KFold(n_splits=5, random_state=7)
results = cross_val_score(adamodel, X, y, cv=kfold)
print("AdaBoost Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

results = cross_val_score(xgbmodel, X, y, cv=kfold)
print("XGBoost Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

results = cross_val_score(rfmodel, X, y, cv=kfold)
print("Random Forests Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
