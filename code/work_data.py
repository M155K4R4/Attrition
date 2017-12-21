import numpy as np
import pandas as pd
import fancyimpute

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
    Replace the columns in {columns} with their las {n} characters converted to int.
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


def preprocess_client(my_df):
    str_variables = str_variables_with_int()
    output = extract_last_n_from_df(my_df, str_variables, 2)
    output = pd.get_dummies(output, drop_first=True)
    #output = output.dropna(axis=1)
    output.set_index('ID_CORRELATIVO', inplace=True)
    return output

def preprocess_client_test(my_df):
    str_variables = str_variables_with_int()
    output = extract_last_n_from_df(my_df, str_variables, 2)
    output = pd.get_dummies(output, drop_first=True)
    return output


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