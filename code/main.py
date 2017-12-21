import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import scale
from sklearn.decomposition import pca
import fancyimpute

import config
import work_data
import models


def main():
    np.random.seed(42)
    logger = config.config_logger(__name__, 10)
    t0 = time.time()

    train_client_path = './data/raw/csv/train_clientes.csv'
    train_reque_path = './data/raw/csv/train_requerimientos.csv'
    test_client_path = './data/raw/csv/test_clientes.csv'
    test_reque_path = './data/raw/csv/test_requerimientos.csv'
    output_path = './output/'
    write_impute_test = True

    logger.info('Beginning execution')
    logger.info('Load dataframes')
    test_client = pd.read_csv(test_client_path, header=0)
    main_client = pd.read_csv(train_client_path, header=0)
    main_reque = pd.read_csv(train_reque_path, header=0)

    work_data.basic_descriptive(main_client)
    work_data.basic_descriptive(main_reque)

    id_variables = work_data.id_variables()
    index_client = test_client['ID_CORRELATIVO']

    if write_impute_test:
        test_client = work_data.preprocess_client_test(test_client)
        test_client = work_data.remove_columns(test_client, id_variables)
        test_client = pd.DataFrame(fancyimpute.IterativeSVD().complete(test_client))
        test_client.to_csv('./data/mod/test_imputed.csv', header=0, index=False)
    else:
        test_client = pd.read_csv('./data/mod/test_imputed.csv', header=None)


    #----- MERGE ----
    #main_reque = pd.get_dummies(main_reque)
    #main_reque = pd.pivot_table(main_reque, index=['ID_CORRELATIVO'],columns=['CODMES'], aggfunc=np.sum)
    #main_reque.columns = main_reque.columns.map('{0[0]}|{0[1]}'.format)
    #main_reque.fillna(0, inplace=True)

    #main_df = pd.concat([main_client, main_reque], axis=1, join_axes=[main_client.index]).reset_index()
    #main_df.fillna(0, inplace=True)
    #print(main_df.head().to_string())
    #work_data.basic_descriptive(main_df)
    #---------------

    #check_var = 'CODMES'
    #print(main_client[check_var].value_counts())

    main_client = work_data.preprocess_client(main_client)
    main_client = main_client.reset_index()
    main_client = work_data.remove_columns(main_client, id_variables)


    #print(main_client.head().to_string())
    #print(main_reque.head().to_string())
    #print(main_client.info())
    #print(main_client.describe().transpose().to_string())

    #y = main_df.pop('ATTRITION')
    #x = main_df
    y = main_client.pop('ATTRITION')
    x = main_client

    logger.info('Replacing missing values')
    x = pd.DataFrame(fancyimpute.IterativeSVD().complete(x))

    #work_data.basic_descriptive(x)
    #hi

    logger.info('Split data into train and test')
    x_train, x_test, y_train, y_test = models.split_data(x, y)
    work_data.basic_descriptive(x_train)

    logger.info('Run models')

    logger.info('XgBoost')
    xgboost_model = models.xgboost_grid(x_train, y_train)
    print(xgboost_model.score(x_test, y_test))

    #logger.info('LGBM')
    #lgbm_model = models.lightgbm_grid(x_train, y_train)
    #print(lgbm_model.score(x_test, y_test))

    logger.info('GBM')
    gbm_model = models.gbm_grid(x_train, y_train)
    print('Test grid: {0}'.format(gbm_model.score(x_test, y_test)))

    #full_gbm_model = models.gbm_full(x_train, y_train)
    #y_test_pred = full_gbm_model.predict_proba(x_test)[:, 1]
    #print('Test full: {0}'.format(log_loss(y_test, y_test_pred)))
    #y_pred = full_gbm_model.predict_proba(test_client)[:, 1]
    #y_pred = pd.Series(y_pred)
    #print(y_pred.shape)

    #final = pd.concat([index_client, y_pred], axis=1, ignore_index=True)
    #final.columns = ['ID_CORRELATIVO', 'ATTRITION']
    #final.to_csv(output_path + 'results_prelim2.csv', index=False)

    #logger.info('Logit')
    #logit_model = models.logit_grid(x_train, y_train)
    #print(logit_model.score(x_test, y_test))

    #logger.info('AdaBoost')
    #ada_model = models.adaboost_grid(x_train, y_train)
    #print(ada_model.score(x_test, y_test))

    config.time_taken_display(t0)

    #TODO Unbalanced classification problem

    # Brain storm for preprocessing:
    # 1. As you see it.
    # 2. Mean across the 5 months.
    # 3. Weighted mean across the 5 months. Give more weight to the most recent.
    # 4. PCA. -> visualize 2 dim!
    # 5. 2-month window average.
    # 6. Discretize continuous variables. Especially those with many zeros.

    # There are some missing values. We must try the following:
    # YES 1. Keep only obs with complete information.
    # NO 2. Input values to missing obs.

    # Fist, create the classification algorithm, then implement the preprocessing alternatives

    # Brain storm for classification:
    # *1. GBM - LGBM
    # 2. XGBoost
    # *3. Adaboost
    # 4. SMOTE
    # 5. Boosting for unbalanced classes.
    # *6. Logit lasso/ridge
    # 7. NN -> problem optimizing, I don't have enough computational power.
    # 8. Voting Classifier - soft (for the best 3?)

    # All the algorithms must be implemented with a 10-fold CV and a GridSearch.
    # Report accuracy and auc


if __name__ == '__main__':
    main()
