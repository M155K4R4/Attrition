import numpy as np
import pandas as pd
import fancyimpute
from sklearn.decomposition import PCA

import config

logger = config.config_logger(__name__, 10)


def basic_descriptive(my_df):
    """
    Report basic descriptive stats for a pd.DataFrame.
    Args:
        my_df (pd.DataFrame): target DataFrame.

    Returns:
        Nothing. All results are printed.
    """
    n_row, n_col = my_df.shape
    cols = my_df.columns
    logger.info('# observations: {0}'.format(n_row))
    logger.info('# features: {0}'.format(n_col))
    logger.info('Features: {0}'.format(cols))
    return


def last_n_to_int(my_str, n):
    """
    Extract the last n characters of a string and convert them to int.
    Args:
        my_str (str): origin of the characters to be extracted
        n (int): amount of characters to be extracted.

    Returns:
        np.nan: if my_str is null.
        int: characters extracted.

    Raises:
        ValueError if las n digits are not digits.
    """
    if pd.isnull(my_str):
        return np.nan
    else:
        output = my_str[-n:]
        if output.isdigit():
            return int(my_str[-n:])
        else:
            raise ValueError('Last n digits are not digits: {0}'.format(my_str))


def extract_last_n_from_df(my_df, columns, n):
    """
    Replace the columns in {columns} with their last {n} characters converted to int.
    Args:
        my_df (pd.DataFrame): target DataFrame. Columns will be extracted from it.
        columns (list): contains target column names.
        n (int): number of characters to be extracted.

    Returns:
        pd.DataFrame: target columns have been replace with the last two characters
            of each observation converted to int.
    """
    my_df = my_df.copy()
    for column in columns:
        target = my_df.pop(column)
        target.replace('Cero', '00', inplace=True)
        my_df[column] = target.apply(lambda x: last_n_to_int(x, n))
    return my_df


def remove_columns(my_df, columns):
    """
    Remove columns from my_df.
    Args:
        my_df (pd.DataFrame): dataframe which columns will be removed.
        columns (list): contains the  names of the columns to be removed.

    Returns:
        pd.DataFrame: dataframe with the columns removed.
    """
    my_df = my_df.copy()
    for column in columns:
        my_df.pop(column)
    return my_df


def continuous_to_discrete(my_df, name, threshold):
    my_df = my_df.copy()
    my_df[name] = pd.Series(list(my_df[name] > threshold), dtype='int32', index=my_df.index)
    return my_df


def preprocess_reque(my_df):
    my_df = my_df.copy()
    del(my_df['PRODUCTO_SERVICIO_2'])
    del(my_df['SUBMOTIVO_2'])
    del(my_df['CODMES'])
    output = pd.get_dummies(my_df)
    output = output.groupby('ID_CORRELATIVO').sum()

    for my_var in discretize_vars():
        output = continuous_to_discrete(output, my_var, 0)
    return output


def preprocess_client(my_df):
    str_variables = str_variables_with_int()
    output = extract_last_n_from_df(my_df, str_variables, 2)
    output = pd.get_dummies(output, drop_first=True)
    id_client = my_df['ID_CORRELATIVO']
    output = remove_columns(output, id_variables())
    client_cols = output.columns
    output = pd.DataFrame(fancyimpute.IterativeSVD(verbose=False).complete(output))
    output.columns = client_cols
    output.index = id_client
    return output


def do_pca(my_df):
    my_df = my_df.copy()
    # Number of components that explain 1% or more of the variance
    pca = PCA()
    pca.fit(my_df)
    print(pca.components_.shape)
    print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_>0.01))
    print(sum(pca.singular_values_ >= 1))
    my_df = pca.transform(my_df)
    print(my_df.shape)
    return my_df


def str_variables_with_int():
    """
    Columns with observations which last two characters are digits. We will extract these.
    Returns:
        list: contains the column names.
    """
    return ['RANG_INGRESO', 'RANG_SDO_PASIVO_MENOS0', 'RANG_NRO_PRODUCTOS_MENOS0']


def id_variables():
    """
    Columns used as identificators.
    Returns:
        list: contains the column names.
    """
    return ['ID_CORRELATIVO', 'CODMES']


def discretize_vars():
    return ['TIPO_REQUERIMIENTO2_Reclamo', 'TIPO_REQUERIMIENTO2_Solicitud', 'DICTAMEN_NO PROCEDE',
            'DICTAMEN_PROCEDE PARCIAL', 'DICTAMEN_PROCEDE TOTAL']