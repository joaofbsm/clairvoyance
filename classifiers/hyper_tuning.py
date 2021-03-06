#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xgboost
import global_constants as gc
from numpy import loadtxt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

dataset_name = sys.argv[1]

# load data
dataset = loadtxt(dataset_name + ".csv", delimiter=",")
input_size = dataset.shape[1] - 1

# split data into X and y
X = dataset[:,0:input_size]
y = dataset[:,input_size]

# fit model no training data
model = xgboost.XGBClassifier()
#model = RandomForestClassifier(n_estimators=100)

params = {
    "learning_rate": [0.001, 0.01, 0.1, 0.3],
    "n_estimators": [50, 100, 1000],
    "max_depth": [2, 4, 6, 8]
}

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, params, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))