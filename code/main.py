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
    do_merge = True
    write_impute_test = True
    write_output = True
    version = 4

    logger.info('Beginning execution')
    logger.info('Load dataframes')
    test_client = pd.read_csv(test_client_path, header=0)
    test_reque = pd.read_csv(test_reque_path, header=0)
    main_client = pd.read_csv(train_client_path, header=0)
    main_reque = pd.read_csv(train_reque_path, header=0)

    work_data.basic_descriptive(main_client)
    work_data.basic_descriptive(main_reque)

    id_variables = work_data.id_variables()
    index_client = test_client['ID_CORRELATIVO']

    if write_impute_test:
        logger.info('Creating new test database')
        logger.info('Cleaning test reque database')
        test_reque = work_data.preprocess_reque(test_reque)
        print(test_reque.head().to_string())

        logger.info('Cleaning test client database - Imputing missing values')
        test_client = work_data.preprocess_client(test_client)
        print(test_client.head().to_string())

        logger.info('Merging test databases')
        temp = pd.concat([test_client, test_reque], axis=1, join_axes=[test_client.index])
        temp.fillna(0, inplace=True)
        test_df = temp
        print(test_df.head().to_string())
        print(test_df.describe().transpose().to_string())

        logger.info('Saving test database')
        test_df.to_csv('./data/mod/test_imputed.csv', index=False)


    else:
        logger.info('Opening test database')
        test_df = pd.read_csv('./data/mod/test_imputed.csv', header=0)
        print(test_df.head().to_string())

    if do_merge:
        logger.info('Creating new merge')
        logger.info('Cleaning reque database')
        main_reque = work_data.preprocess_reque(main_reque)
        print(main_reque.head().to_string())

        #main_reque = pd.pivot_table(main_reque, index=['ID_CORRELATIVO'], columns=['CODMES'], aggfunc=np.sum)
        #main_reque.columns = main_reque.columns.map('{0[0]}|{0[1]}'.format)
        #main_reque.fillna(0, inplace=True)

        logger.info('Cleaning client database - Imputing missing values')
        target = main_client.pop('ATTRITION')
        target.index = main_client['ID_CORRELATIVO']
        main_client = work_data.preprocess_client(main_client)
        main_client['ATTRITION'] = target
        print(main_client.head().to_string())

        logger.info('Merging databases')
        temp = pd.concat([main_client, main_reque], axis=1, join_axes=[main_client.index])
        temp.fillna(0, inplace=True)
        main_df = temp


        #logger.info('Cleaning reque')
        #temp_reque = temp[reque_cols].copy()
        #temp_reque.fillna(0, inplace=True)

        #logger.info('Implementing PCA')
        #temp_reque = pd.DataFrame(work_data.do_pca(temp_reque))
        #temp_reque.index = temp_index
        #pca_cols = ['comp_{0}'.format(x) for x in temp_reque.columns]
        #main_client[pca_cols] = temp_reque
        #main_df = main_client

        print(main_df.shape)
        print(main_df.head().to_string())
        print(main_df.describe().transpose().to_string())
        work_data.basic_descriptive(main_df)

        logger.info('Saving marges database')
        main_df.to_csv('./data/mod/merge1.csv', index=False)
    else:
        logger.info('Opening merged database')
        main_df = pd.read_csv('./data/mod/merge1.csv', header=0)
        print(main_df.head().to_string())
        print(main_df.shape)


    #check_var = 'CODMES'
    #print(main_client[check_var].value_counts())

    y = main_df.pop('ATTRITION')
    x = main_df

    logger.info('Split data into train and test')
    x_train, x_test, y_train, y_test = models.split_data(x, y)
    work_data.basic_descriptive(x_train)

    logger.info('Run models')

    #logger.info('XgBoost')
    #xgboost_model = models.xgboost_grid(x_train, y_train)
    #print('Test grid: {0}'.format(xgboost_model.score(x_test, y_test)))
    #Test: -0.326

    #logger.info('LGBM')
    #lgbm_model = models.lightgbm_grid(x_train, y_train)
    #print(lgbm_model.score(x_test, y_test))

    logger.info('GBM')
    gbm_model = models.gbm_grid(x_train, y_train)
    print('Test grid: {0}'.format(gbm_model.score(x_test, y_test)))
    #Test: -0.314

    if write_output:
        full_gbm_model = models.gbm_full(x_train, y_train)
        y_test_pred = full_gbm_model.predict_proba(x_test)[:, 1]
        print('Test full: {0}'.format(log_loss(y_test, y_test_pred)))
        y_pred = full_gbm_model.predict_proba(test_df)[:, 1]
        y_pred = pd.Series(y_pred)
        print(y_pred.shape)

        final = pd.concat([index_client, y_pred], axis=1, ignore_index=True)
        final.columns = ['ID_CORRELATIVO', 'ATTRITION']
        final.to_csv(output_path + 'results_prelim{0].csv'.format(version), index=False)

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
