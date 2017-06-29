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

#pyplot.style.use("bmh")

dataset_name = sys.argv[1]

# load data
dataset = loadtxt(dataset_name + ".csv", delimiter=",")
input_size = dataset.shape[1] - 1

# split data into X and y
X = dataset[:,0:input_size]
y = dataset[:,input_size]

# split data into train and test sets
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

xgboost.plot_importance(model)
print(model.feature_importances_)
pyplot.show()