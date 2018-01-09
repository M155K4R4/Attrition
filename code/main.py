import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import scale
from sklearn.decomposition import pca
import fancyimpute
from sklearn.preprocessing import StandardScaler
import xgbfir
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

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
    do_merge = False
    write_impute_test = False
    write_output = False
    add_variables = False
    version = 5

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
        test_client = work_data.count_missings_column(test_client)
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
        main_client = work_data.count_missings_column(main_client)
        target = main_client.pop('ATTRITION')
        target.index = main_client['ID_CORRELATIVO']
        main_client = work_data.preprocess_client(main_client)
        main_client['ATTRITION'] = target
        print(main_client.head().to_string())

        logger.info('Merging databases')
        temp = pd.concat([main_client, main_reque], axis=1, join_axes=[main_client.index])
        temp.fillna(0, inplace=True)
        main_df = temp

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

    y = main_df.pop('ATTRITION')
    main_df = main_df.append(test_df).reset_index(drop=True)

    if False:
        logger.info('Creating T-SNE database')
        temp_tsne = pd.DataFrame(models.tnse(main_df))
        temp_tsne.to_csv('./data/mod/merge1_tsne.csv', index=False)
    else:
        logger.info('Loading T-SNE database')
        temp_tsne = pd.read_csv('./data/mod/merge1_tsne.csv')

    if add_variables:
        logger.info('Beginning feature engineering')
        logger.info('Interactions')
        main_df_feat = models.create_interactions(main_df, models.inter_vars())

        logger.info('Row sums 1-3')
        main_df_feat['ext1'] = main_df.apply(lambda row: (row == 0).sum(), axis=1)
        temp = models.standard_scale_df(main_df)
        main_df_feat['ext2'] = temp.apply(lambda row: (row > 0.5).sum(), axis=1)
        main_df_feat['ext3'] = temp.apply(lambda row: (row < -0.5).sum(), axis=1)

        logger.info('K-means 4-7')
        main_df_feat['ext4'] = pd.Series(models.kmeans(main_df, 5)).apply(str)
        main_df_feat['ext5'] = pd.Series(models.kmeans(main_df, 10)).apply(str)
        main_df_feat['ext6'] = pd.Series(models.kmeans(main_df, 15)).apply(str)
        main_df_feat['ext7'] = pd.Series(models.kmeans(main_df, 20)).apply(str)

        logger.info('KNN 8-11')
        main_df_feat['ext8'] = models.knn_distance(main_df, 2)
        main_df_feat['ext9'] = models.knn_distance(main_df, 3)
        main_df_feat['ext10'] = models.knn_distance(main_df, 5)
        main_df_feat['ext11'] = models.knn_distance(temp_tsne, 2)

        main_df_feat = pd.get_dummies(main_df_feat, drop_first=True)
        print(main_df_feat.head().to_string())
        print(main_df_feat.shape)
        config.time_taken_display(t0)
        logger.info('Saving features database')
        main_df_feat.to_csv('./data/mod/merge1_features.csv', index=False)
    else:
        logger.info('Opening feature engineered database')
        main_df_feat = pd.read_csv('./data/mod/merge1_features.csv', header=0)
        print(main_df_feat.head().to_string())
        print(main_df_feat.shape)

    logger.info('Split data into train and test')
    x, test_df = main_df_feat.iloc[:70000, :], main_df_feat.iloc[70000:, :]
    print(main_df_feat.shape)
    print(x.shape)
    print(test_df.shape)
    x_train, x_test, y_train, y_test = models.split_data(x, y)
    work_data.basic_descriptive(x_train)

    logger.info('Level 1 - Create metafeatures')

    if False:
        logger.info('1. Ridge logit')
        ridge_model = models.logit_grid(x, y, 'l2', StandardScaler())
        models.write_prediction(ridge_model, main_df_feat, index_client, 'ridge_standard')
        # print(ridge_model.score(x_test, y_test))

        logger.info('2. Lasso logit')
        lasso_model = models.logit_grid(x, y, 'l1',StandardScaler())
        models.write_prediction(lasso_model, main_df_feat, index_client, 'lasso_standard')
        # print(lasso_model.score(x_test, y_test))

        logger.info('3. Random Forrest')
        RF_model = models.random_forrest_grid(x, y, StandardScaler())
        models.write_prediction(RF_model, main_df_feat, index_client, 'RF_standard')

        logger.info('4. Extra Trees')
        ET_model = models.extra_trees_grid(x, y, StandardScaler())
        models.write_prediction(ET_model, main_df_feat, index_client, 'ET_standard')

    logger.info('5. 2-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 2)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN2_standard')

    logger.info('6. 4-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 4)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN4_standard')

    logger.info('7. 8-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 8)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN8_standard')

    logger.info('8. 16-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 16)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN16_standard')

    logger.info('9. 32-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 32)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN32_standard')

    logger.info('10. 64-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 64)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN64_standard')

    logger.info('11. 128-KNN')
    KNN_model = models.knn_grid(x, y, StandardScaler(), 128)
    models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN128_standard')

    config.time_taken_display(t0)
    hi

    logger.info('XgBoost')
    xgboost_result = models.xgboost_grid(x_train, y_train, x_test, y_test)
    print('Test grid: {0}'.format(xgboost_result))
    #Test: -0.322

    xgboost_full = models.xgboost_full_mod(x_train, y_train, x_test, y_test)
    print(xgboost_full)
    xgbfir.saveXgbFI(xgboost_full, feature_names=main_df.columns, OutputXlsxFile='./data/mod/bbva.xlsx')

    #xgboost_full_result = models.xgboost_full(x_train, y_train, x_test, y_test)
    #print('Test grid: {0}'.format(xgboost_full_result))

    #logger.info('LGBM')
    #lgbm_model = models.lightgbm_grid(x_train, y_train)
    #print(lgbm_model.score(x_test, y_test))

    #logger.info('GBM')
    #gbm_model = models.gbm_grid(x_train, y_train)
    #print('Test grid: {0}'.format(gbm_model.score(x_test, y_test)))
    #Test: -0.314

    if write_output:
        logger.info('Predict test')
        y_pred = xgboost_full.predict_proba(test_df)[:, 1]
        y_pred = pd.Series(y_pred)
        print(y_pred.shape)

        logger.info('Saving predictions')
        final = pd.concat([index_client, y_pred], axis=1, ignore_index=True)
        final.columns = ['ID_CORRELATIVO', 'ATTRITION']
        final.to_csv(output_path + 'results_prelim{0}.csv'.format(version), index=False)

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
