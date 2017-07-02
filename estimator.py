import sys
import xgboost
import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series(dict(params.items() + d.items()))

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


def main():

    dataset_name = sys.argv[1]

    # load data
    dataset = loadtxt(dataset_name + ".csv", delimiter=",")
    input_size = dataset.shape[1] - 1

    # split data into X and y
    X = dataset[:,0:input_size]
    y = dataset[:,input_size]

    models1 = { 
        'AdaBoostClassifier': AdaBoostClassifier(),
        'XGBoost': xgboost.XGBClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    params1 = { 
        'AdaBoostClassifier':  {'n_estimators': [50, 100, 200],
                                'learning_rate': [0.01, 0.1, 1]},
        'XGBoost': {'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'max_depth': [4, 6, 8, 10]},
        'RandomForestClassifier': {'n_estimators': [50, 100, 200],
                                   'max_depth': [4, 6, 8, 10]}
    }

    helper1 = EstimatorSelectionHelper(models1, params1)
    helper1.fit(X, y, scoring='f1', cv=5, n_jobs=-1)

    print(helper1.score_summary(sort_by='min_score'))


if __name__ == '__main__':
    main()