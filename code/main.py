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
    print(main_client.head().to_string())

    check_var = 'ATTRITION'
    print(main_client[check_var].value_counts())

    str_variables = work_data.str_variables_with_int()
    for i in str_variables:
        print(main_client[i].value_counts())

    main_client = work_data.extract_last_n_from_df(main_client, str_variables, 2)

    print(main_client.head().to_string())
    print(main_reque.head().to_string())
    print(main_client.info())
    print(main_client.describe().transpose().to_string())

    config.time_taken_display(t0)

    #TODO Unbalanced classification problem

    # Brain storm for preprocessing:
    # 1. As you see it.
    # 2. Mean across the 5 months.
    # 3. Weighted mean across the 5 months. Give more weight to the most recent.
    # 4. PCA. -> visualize 2 dim!
    # 5. 2-month window average.
    # 6. Discretize continuous variables. Especially those with many zeros.

    # Fist, create the classification algorithm, then implement the preprocessing alternatives

    # Brain storm for classification:
    # 1. GBM
    # 2. XGBoost
    # 3. Adaboost
    # 4. SMOTE
    # 5. Boosting for unbalanced classes.
    # 6. Logit lasso/ridge
    # 7. NN -> problem optimizing
    # 8. Voting Classifier - soft (for the best 3?)

    # All the algorithms must be implemented with a 10-fold CV and a GridSearch.


if __name__ == '__main__':
    main()
