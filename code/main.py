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
import os


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
    version = 6

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
        print(ridge_model.score(x_test, y_test))

        logger.info('2. Lasso logit')
        lasso_model = models.logit_grid(x, y, 'l1',StandardScaler())
        models.write_prediction(lasso_model, main_df_feat, index_client, 'lasso_standard')
        print(lasso_model.score(x_test, y_test))

        logger.info('3. Random Forrest')
        RF_model = models.random_forrest_grid(x, y, StandardScaler())
        models.write_prediction(RF_model, main_df_feat, index_client, 'RF_standard')
        print(RF_model.score(x_test, y_test))

        logger.info('4. Extra Trees')
        ET_model = models.extra_trees_grid(x, y, StandardScaler())
        models.write_prediction(ET_model, main_df_feat, index_client, 'ET_standard')
        print(ET_model.score(x_test, y_test))

        logger.info('5. 2-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 2)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN2_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('6. 4-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 4)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN4_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('7. 8-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 8)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN8_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('8. 16-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 16)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN16_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('9. 32-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 32)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN32_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('10. 64-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 64)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN64_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('11. 128-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 128)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN128_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('12. 256-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 256)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN256_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('13. 512-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 512)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN512_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('14. 1024-KNN')
        KNN_model = models.knn_grid(x, y, StandardScaler(), 1024)
        models.write_prediction(KNN_model, main_df_feat, index_client, 'KNN1024_standard')
        print(KNN_model.score(x_test, y_test))

        logger.info('15. Naive Bayes')
        NB_model = models.naive_bayes_grid(x, y, StandardScaler())
        models.write_prediction(NB_model, main_df_feat, index_client, 'NB_standard')
        print(NB_model.score(x_test, y_test))

        logger.info('16. MPL')
        MLP_model = models.MLP_grid(x, y, StandardScaler())
        models.write_prediction(MLP_model, main_df_feat, index_client, 'MLP_standard')
        print(MLP_model.score(x_test, y_test))

        logger.info('17. AdaBoost')
        adaboost_model = models.adaboost_grid(x, y, StandardScaler())
        models.write_prediction(adaboost_model, main_df_feat, index_client, 'adaboost_standard')
        print(adaboost_model.score(x_test, y_test))

        logger.info('18. GBM')
        gbm_model = models.gbm_grid(x, y, StandardScaler())
        models.write_prediction(gbm_model, main_df_feat, index_client, 'gbm_standard')
        print(gbm_model.score(x_test, y_test))

        logger.info('18. LightGBM')
        lgbm_model = models.lgbm_grid(x, y, None)
        models.write_prediction(lgbm_model, main_df_feat, index_client, 'lgbm_none')
        print(lgbm_model.score(x_test, y_test))

    logger.info('19. XgBoost')
    test_final = main_df_feat.iloc[70000:, :]
    id_test = test_client['ID_CORRELATIVO']
    xgboost_model = models.xgboost_grid(x, y, StandardScaler())
    models.write_prediction(xgboost_model, main_df_feat, index_client, 'xgboost_standard')
    print(xgboost_model.score(x_test, y_test))
    models.write_prediction(xgboost_model, test_final, id_test, 'ATTRITION')
    hi

    # Stage 2:
    logger.info('Level 2')
    logger.info('Creating meta-features database')
    meta_features_list = os.listdir('./data/mod/meta_features')
    temp = {}
    for feature in meta_features_list:
        temp_df = pd.read_csv('./data/mod/meta_features/{0}'.format(feature), header=0)
        temp[feature] = temp_df.iloc[:, 1]
    meta_features = pd.DataFrame(temp)
    meta_features = pd.concat([meta_features, main_df_feat], axis=1, ignore_index=True)
    x = meta_features.iloc[:70000, :]
    test_final = meta_features.iloc[70000:, :]
    x_train, x_test, y_train, y_test = models.split_data(x, y)

    print(x_train.shape)
    print(test_final.shape)
    print(x.shape)

    logger.info('Estimating second level model with XgBoost')
    xgboost_final = models.xgboost_full_mod(x_train, y_train)
    print(xgboost_final.score(x_test, y_test))
    print(models.get_logloss(y_test, xgboost_final.predict_proba(x_test)[:, 1]))
    models.write_final_prediction(xgboost_final, test_final, test_client['ID_CORRELATIVO'], 'results8')
    models.write_final_prediction(xgboost_final, x, main_client['ATTRITION'], 'train')


    config.time_taken_display(t0)
    hi

    logger.info('XgBoost')
    xgboost_result = models.xgboost_grid(x_train, y_train, x_test, y_test)
    print('Test grid: {0}'.format(xgboost_result))
    #Test: -0.322

    xgboost_full = models.xgboost_full_mod(x_train, y_train, x_test, y_test)
    print(xgboost_full)
    xgbfir.saveXgbFI(xgboost_full, feature_names=main_df.columns, OutputXlsxFile='./data/mod/bbva.xlsx')

if __name__ == '__main__':
    main()
