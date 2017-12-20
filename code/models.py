import numpy as np
import pprint
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

import config

logger = config.config_logger(__name__,10)
np.random.seed(42)


def split_data(x, y):
    return train_test_split(x, y, test_size=0.4)


def model_fit(alg, name_alg, params, x, y):
    steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
             ('scale', StandardScaler()),
             (name_alg, alg)]
    pipeline = Pipeline(steps)
    alg_cv = GridSearchCV(pipeline, params, cv=10, n_jobs=7)
    alg_cv.fit(x, y)
    pprint.pprint(alg_cv.cv_results_)
    print(alg_cv.best_score_)
    print(alg_cv.best_params_)
    return alg_cv


def gbm_grid(x, y):
    alg = GradientBoostingClassifier(n_estimators=100)
    alg_name = 'gbm'
    params = {}
    model = model_fit(alg, alg_name, params, x, y)
    return model


def logit_grid(x, y):
    alg = LogisticRegression()
    alg_name = 'logit'
    params = {'logit__C': np.logspace(-3, 1, num=10),
              'logit__penalty': ['l1', 'l2']}
    model = model_fit(alg, alg_name, params, x, y)
    return model


def adaboost_grid(x, y):
    alg = AdaBoostClassifier()
    alg_name = 'ada'
    params = {}
    model = model_fit(alg, alg_name, params, x, y)
    return model