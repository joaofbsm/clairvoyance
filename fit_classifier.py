#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import xgboost
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

dataset_input_size = {"pre1": 272,
                      "pre2": 290,
                      "pre5": 278,
                      "pre6": 276,
                      "pre7": 274,
                      "pre8": 274,
                      "pre9": 282,
                      "prein1": 280,
                      "prein1diff": 284}

input_size = dataset_input_size[dataset_name]

# split data into X and y
X = dataset[:,0:input_size]
y = dataset[:,input_size]

# split data into train and test sets
seed = 7
test_size = 0.33
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
pyplot.show()