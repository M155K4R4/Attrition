import numpy as np
import pandas as pd
import pprint
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, \
    RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, scale, FunctionTransformer, OneHotEncoder, \
    PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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
    return train_test_split(x, y, test_size=0.4)


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


def gbm_grid(x, y, scaling):
    alg = GradientBoostingClassifier(n_estimators=300)
    alg_name = 'gbm'
    params = {'gbm__min_samples_split': [3],
              'gbm__min_samples_leaf': [5]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def logit_grid(x, y, penalty, scaling):
    alg = LogisticRegression()
    alg_name = 'logit'
    params = {'logit__C': np.logspace(-3, 1, num=10),
              'logit__penalty': [penalty]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def random_forrest_grid(x, y, scaling):
    alg = RandomForestClassifier(n_estimators=300)
    alg_name = 'RF'
    params = {'RF__min_samples_split': [3],
              'RF__min_samples_leaf': [5]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def extra_trees_grid(x, y, scaling):
    alg = ExtraTreesClassifier(n_estimators=300)
    alg_name = 'ET'
    params = {'ET__min_samples_split': [3],
              'ET__min_samples_leaf': [5],
              'ET__max_features': [22]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def knn_grid(x, y, scaling, n):
    alg = KNeighborsClassifier()
    alg_name = 'KNN'
    params = {'KNN__n_neighbors': [n],
              'KNN__p': [1, 2]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def naive_bayes_grid(x, y, scaling):
    alg = GaussianNB()
    alg_name = 'NB'
    params = {}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def MLP_grid(x, y, scaling):
    alg = MLPClassifier(activation='relu', max_iter=200)
    alg_name = 'MLP'
    params = {'MLP__hidden_layer_sizes':[(200, 100)],
              'MLP__alpha': [0.077426368268112777],
              'MLP__max_iter': [250]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def adaboost_grid(x, y, scaling):
    alg = AdaBoostClassifier()
    alg_name = 'ada'
    params = {'ada__learning_rate': [0.3],
              'ada__n_estimators': [200]}
    model = grid_fit(alg, alg_name, params, x, y, scaling)
    return model


def xgboost_grid(x, y, scaling):
    if False:
        x_temp = scale(x)
        x_dmatrix = xgb.DMatrix(x_temp, label=y)
        params = {'objective': 'binary:logistic',
                  'silent': 1,
                  'eval_metric': 'logloss',
                  'max_depth': 10,
                  'colsample_bytree': 0.8,
                  'eta': 0.1,
                  'min_child_weight': 0.3,
                  'subsample': 1}
        cv_results = xgb.cv(dtrain=x_dmatrix,
                            params=params,
                            nfold=5,
                            num_boost_round=1000,
                            metrics="logloss",
                            as_pandas=True,
                            seed=42,
                            early_stopping_rounds=250)
        print(cv_results)
    params_grid = {}
    alg = xgb.XGBClassifier(objective='binary:logistic',
                            silent=1,
                            max_depth=10,
                            colsample_bytree=0.8,
                            learning_rate=0.1,
                            min_child_weight=0.3,
                            subsample=1,
                            n_estimators=51)
    #n_estimators = 99
    model = grid_fit(alg, 'xgb', params_grid, x, y, scaling)
    return model


def xgboost_full_mod(x, y):
    alg = xgb.XGBClassifier(objective='binary:logistic',
                            silent=1,
                            max_depth=3,
                            colsample_bytree=0.8,
                            learning_rate=0.5,
                            min_child_weight=0.3,
                            subsample=0.3,
                            n_estimators=99,
                            reg_lambda=200,
                            reg_alpha=100)
    #n_estimators = 99
    alg.fit(x, y, eval_metric='logloss')
    xgb.plot_importance(alg)
    plt.show()
    return alg


def xgboost_full(x, y, x_test, y_test):
    x = scale(x)
    x_dmatrix = xgb.DMatrix(x, label=y)
    x_test = scale(x_test)
    test_dmatrix = xgb.DMatrix(x_test, label=y_test)
    params = {"objective": "binary:logistic",
              'silent': 1,
              "max_depth": 10,
              "colsample_bytree": 0.8,
              "learning_rate": 0.1,
              "min_child_weight": 0.3,
              "subsample": 1,
              "reg_alpha": 50,
              "reg_lambda": 50}
    xg_boost = xgb.train(dtrain=x_dmatrix,
                         params=params,
                         num_boost_round=99,
                         verbose_eval=False)
    xgb.plot_importance(xg_boost)
    plt.show()
    preds = xg_boost.predict(test_dmatrix)
    print('Test grid: {0}'.format(log_loss(y_test, preds)))
    return xg_boost


def lgbm_grid(x, y, scaling):
    grid_params = {}
    alg_name = 'lgbm'
    alg = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             silent=True,
                             n_estimators=300,
                             max_depth=-1,
                             num_leaves=31,
                             learning_rate=0.1,
                             subsample_for_bin=50000,
                             subsample=1,
                             subsample_freq=1,
                             min_split_gain=0.,
                             min_child_samples=20,
                             reg_alpha=0.,
                             reg_lambda=0.)
    model = grid_fit(alg, alg_name, grid_params, x, y, scaling)
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
    final.to_csv('./data/mod/meta_features/{0}.csv'.format(name), index=False)
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


def log_df(my_df):
    output = np.apply_along_axis(sum_min, 0, my_df)
    output = np.log(output + 1)
    return output


def sum_min(my_series):
    output = my_series + np.abs(np.min(my_series))
    return output


def log_transformer():
    return FunctionTransformer(log_df)


def write_final_prediction(model, x, index, name):
    y_pred = model.predict_proba(x)[:, 1]
    y_pred = pd.Series(y_pred)
    final = pd.concat([index, y_pred], axis=1, ignore_index=True)
    final.columns = ['ID_CORRELATIVO', 'ATTRITION']
    final.to_csv('./output/{0}.csv'.format(name), index=False)
    return


def get_logloss(y, y_pred):
    return log_loss(y, y_pred)
