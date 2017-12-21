import numpy as np
import pprint
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

import config

logger = config.config_logger(__name__,10)
np.random.seed(42)


def split_data(x, y):
    return train_test_split(x, y, test_size=0.3)


def model_fit(alg, name_alg, x, y):
    steps = [('scale', StandardScaler()),
             (name_alg, alg)]
    pipeline = Pipeline(steps)
    pipeline.fit(x, y)
    return pipeline


def gbm_full(x, y):
    alg = GradientBoostingClassifier(n_estimators=300)
    alg_name = 'gbm'
    model = model_fit(alg, alg_name, x, y)
    return model


def grid_fit(alg, name_alg, params, x, y):
    steps = [('scale', StandardScaler()),
             (name_alg, alg)]
    pipeline = Pipeline(steps)
    alg_cv = GridSearchCV(pipeline, params, cv=10, n_jobs=8, scoring='neg_log_loss')
    alg_cv.fit(x, y)
    pprint.pprint(alg_cv.cv_results_)
    print(alg_cv.best_score_)
    print(alg_cv.best_params_)
    return alg_cv


def gbm_grid(x, y):
    alg = GradientBoostingClassifier(n_estimators=300)
    alg_name = 'gbm'
    params = {}
    model = grid_fit(alg, alg_name, params, x, y)
    return model


def logit_grid(x, y):
    alg = LogisticRegression()
    alg_name = 'logit'
    params = {'logit__C': np.logspace(-3, 1, num=10),
              'logit__penalty': ['l1', 'l2']}
    model = grid_fit(alg, alg_name, params, x, y)
    return model


def adaboost_grid(x, y):
    alg = AdaBoostClassifier()
    alg_name = 'ada'
    params = {}
    model = grid_fit(alg, alg_name, params, x, y)
    return model


def xgboost_grid(x, y):
    alg = xgb.XGBClassifier(objective='binary:logistic')
    alg_name = 'xgboost'
    params = {}
    model = grid_fit(alg, alg_name, params, x, y)
    return model


def lightgbm_grid(x, y):
    #lgb.Dataset(x, y, max_bin=512)
    params = {
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'objective': 'binary',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'subsample_for_bin': 200,
        'subsample': 1,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 5,
        'reg_lambda': 10,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 5,
        'scale_pos_weight': 1,
        'num_class': 1,
        'metric': ['binary_error', 'auc']
    }
    grid_params = {}
    alg_name = 'lgbm'
    alg = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             silent=True,
                             max_depth=params['max_depth'],
                             subsample_for_bin=params['subsample_for_bin'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             min_split_gain=params['min_split_gain'],
                             min_child_weight=params['min_child_weight'],
                             min_child_samples=params['min_child_samples'],
                             scale_pos_weight=params['scale_pos_weight'])
    model = grid_fit(alg, alg_name, grid_params, x, y)
    return model


def voting(algorithms, x, y):
    vote = VotingClassifier(estimators=algorithms, voting='soft')
    model = vote.fit(x, y)
    return model
