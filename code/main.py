import time
import numpy as np
import pandas as pd

import config
import work_data
import models


def main():
    logger = config.config_logger(__name__, 10)
    t0 = time.time()

    train_client_path = './data/raw/csv/train_clientes.csv'
    train_reque_path = './data/raw/csv/train_requerimientos.csv'
    test_client_path = './data/raw/csv/test_clientes.csv'
    test_reque_path = './data/raw/csv/test_requerimientos.csv'

    logger.info('Beginning execution')
    logger.info('Load dataframes')
    main_client = work_data.load_csv_to_df(train_client_path)
    main_reque = work_data.load_csv_to_df(train_reque_path)

    work_data.basic_descriptive(main_client)
    work_data.basic_descriptive(main_reque)
    #print(main_client.head().to_string())

    #check_var = 'CODMES'
    #print(main_client[check_var].value_counts())

    str_variables = work_data.str_variables_with_int()
    id_variables = work_data.id_variables()

    main_client = work_data.extract_last_n_from_df(main_client, str_variables, 2)
    main_client = pd.get_dummies(main_client, drop_first=True)
    main_client = main_client.dropna()
    main_client = work_data.remove_columns(main_client, id_variables)

    #print(main_client.head().to_string())
    #print(main_reque.head().to_string())
    #print(main_client.info())
    #print(main_client.describe().transpose().to_string())

    y = main_client.pop('ATTRITION')
    x = main_client

    x_train, x_test, y_train, y_test = models.split_data(x, y)
    work_data.basic_descriptive(x_train)

    gbm_model = models.gbm_grid(x_train, y_train)
    print(gbm_model.score(x_test, y_test))

    #logit_model = models.logit_grid(x_train, y_train)
    #print(logit_model.score(x_test, y_test))

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
    # *1. GBM
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
