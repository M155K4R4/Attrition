import numpy as np
import pandas as pd

import config

logger = config.config_logger(__name__, 10)


def load_csv_to_df(path):
    """
    Open a CSV file with headers and a pd.DataFrame.
    Args:
        path (str): path to the CSV file.

    Returns:
        pd.DataFrame: CSV opened.
    """
    return pd.read_csv(path, header=0)


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
