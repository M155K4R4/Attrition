import numpy as np
import pandas as pd
import pprint
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, scale, FunctionTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import config

logger = config.config_logger(__name__,10)
np.random.seed(42)


def model_fit_xgb(alg, x, y, x_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x, y)

    # Predict training set:
    dtrain_predictions = alg.predict(x)
    dtrain_predprob = alg.predict_proba(x)[:, 1]
    dtrain_predictions_test = alg.predict(x_test)
    dtrain_predprob_test = alg.predict_proba(x_test)[:, 1]

    # Print model report:
    print("\nModel Report")
    print('Accuracy (Train): {0}'.format(accuracy_score(y, dtrain_predictions)))
    print('Logloss (Train): {0}'.format(log_loss(y, dtrain_predprob)))
    print('Accuracy (Test): {0}'.format(accuracy_score(y_test, dtrain_predictions_test)))
    print('Logloss (Test): {0}'.format(log_loss(y_test, dtrain_predprob_test)))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance')
    plt.ylabel('Feature Importance Score')
    plt.show()
    return alg


def split_data(x, y):
    return train_test_split(x, y, test_size=0.3)


def model_fit(alg, name_alg, x, y, scaling):
    steps = [('scale', scaling),
             (name_alg, alg)]
    pipeline = Pipeline(steps)
    pipeline.fit(x, y)
    return pipeline


def gbm_full(x, y):
    alg = GradientBoostingClassifier(n_estimators=300)
    alg_name = 'gbm'
    model = model_fit(alg, alg_name, x, y)
    return model


def grid_fit(alg, name_alg, params, x, y, scaling):
    steps = [('scale', scaling),
             (name_alg, alg)]
    pipeline = Pipeline(steps)
    alg_cv = GridSearchCV(pipeline, params, cv=5, n_jobs=8, scoring='neg_log_loss')
    alg_cv.fit(x, y)
    #pprint.pprint(alg_cv.cv_results_)
    logger.info('Best score: {0}'.format(alg_cv.best_score_))
    logger.info('Best params: {0}'.format(alg_cv.best_params_))
    return alg_cv


def gbm_grid(x, y):
    alg = GradientBoostingClassifier(n_estimators=300)
    alg_name = 'gbm'
    params = {}
    model = grid_fit(alg, alg_name, params, x, y)
    return model


def logit_grid(x, y, penalty, scaling):
    alg = LogisticRegression()
    alg_name = 'logit'
    params = {'logit__C': np.logspace(-3, 1, num=10),
              'logit__penalty': [penalty]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def adaboost_grid(x, y):
    alg = AdaBoostClassifier()
    alg_name = 'ada'
    params = {}
    model = grid_fit(alg, alg_name, params, x, y)
    return model


def xgboost_grid(x, y, x_test, y_test):
    x = scale(x)
    x_dmatrix = xgb.DMatrix(x, label=y)
    x_test = scale(x_test)
    test_dmatrix = xgb.DMatrix(x_test, label=y_test)
    params = {'objective': 'binary:logistic', 'silent': 1, 'eval_metric': 'logloss',
              'max_depth': 10, 'colsample_bytree': 0.8, 'eta': 0.2, 'min_child_weight': 0.3, 'subsample': 0.8}
    cv_results = xgb.cv(dtrain=x_dmatrix,
                        params=params,
                        nfold=5,
                        num_boost_round=1000,
                        metrics="logloss",
                        as_pandas=True,
                        seed=42,
                        early_stopping_rounds=50)
    return cv_results


def xgboost_full_mod(x, y, x_test, y_test):
    xgb1 = xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=1000,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=1,
                seed=42)
    alg = model_fit_xgb(xgb1, x, y, x_test, y_test)
    return alg


def xgboost_full(x, y, x_test, y_test):
    x = scale(x)
    x_dmatrix = xgb.DMatrix(x, label=y)
    x_test = scale(x_test)
    test_dmatrix = xgb.DMatrix(x_test, label=y_test)
    params = {"objective": "binary:logistic", 'silent': 1, "max_depth": 3}
    xg_boost = xgb.train(dtrain=x_dmatrix,
                         params=params,
                         num_boost_round=100,
                         verbose_eval=False,
                         early_stopping_rounds=50)
    xgb.plot_importance(xg_boost)
    plt.show()
    preds = xg_boost.predict(test_dmatrix)
    print('Test grid: {0}'.format(log_loss(y_test, preds)))
    return xg_boost



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


def write_prediction(model, x, index, name):
    y_pred = model.predict_proba(x)[:, 1]
    y_pred = pd.Series(y_pred)
    final = pd.concat([index, y_pred], axis=1, ignore_index=True)
    final.columns = ['ID_CORRELATIVO', name]
    final.to_csv('./data/mod/{0}.csv'.format(name), index=False)
    return


def kmeans(my_df, n):
    my_df = my_df.copy()
    model = KMeans(n, n_jobs=-2)
    model.fit(my_df)
    clusters = model.predict(my_df)
    return clusters


def standard_scale_df(my_df):
    return pd.DataFrame(scale(my_df), columns=my_df.columns)


def tnse(my_df):
    output = TSNE(n_components=2, n_jobs=7, perplexity=30, random_state=42).fit_transform(my_df)
    return output


def create_interactions(my_df, my_vars):
    target = my_df[my_vars].copy()
    output = PolynomialFeatures(include_bias=False).fit_transform(target)
    names = list(my_df.columns) + ['inter_{0}'.format(x) for x in range(output.shape[1])]
    output = np.concatenate((my_df, output), axis=1)
    return pd.DataFrame(output, columns=names)


def inter_vars():
    return ['RANG_SDO_PASIVO_MENOS0',
            'ANTIGUEDAD',
            'RANG_INGRESO',
            'RANG_NRO_PRODUCTOS_MENOS0',
            'FLG_SEGURO_MENOS1',
            'NRO_ACCES_CANAL3_MENOS1']


def knn_distance(my_df, n):
    model = NearestNeighbors(n_jobs=7)
    model.fit(my_df)
    output = model.kneighbors(np.array(my_df), n_neighbors=n)
    output = output[0].sum(axis=1)
    return output
