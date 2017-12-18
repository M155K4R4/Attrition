import logging
import time


def config_logger(name, level=10):
    """ Config logger output with level 'level'.

    Args:
        name (str): name of the logger.
        level (int): level of severity displayed.

    Returns:
        object: configured logger.
    """
    logging.basicConfig(level = level, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger(name)
    return logger


def time_in_HMS(begin, finish):
    """ Calculate time difference between begin and finish.

    Args:
        begin (float): time at start
        finish (float): time at finish

    Returns:
        tuple: differences between begin and finish:
            raw difference, hours, minutes and seconds
    """
    raw = finish - begin
    hours = round(raw // 3600)
    minutes = round((raw % 3600) // 60)
    seconds = round((raw % 3600) % 60)
    return raw, hours, minutes, seconds


def time_taken_display(begin, finish=None):
    """ Display in logger how much time has passed between begin and finish.

    Args:
        begin (float): time at start.
        finish (float): time at finish. If None, take current time.

    Returns:
        nothing.
    """
    if finish is None:
        finish = time.time()

    if finish < begin:
        logger.error('Finish time lower than begin time. Begin: {0} - Finish: {1}'.format(begin, finish))
        raise ValueError('Finish time cannot be lower than begin time')

    [check_negative(x) for x in (begin, finish)]

    raw, hours, minutes, seconds = time_in_HMS(begin, finish)
    logger.debug('Execution took a raw time of {0} seconds'.format(round(raw, 5)))
    logger.info('Execution took {0} hours, {1} minutes and {2} seconds'.format(hours, minutes, seconds))
    return


def check_negative(number):
    """ Raise error if number is below zero.

    Args:
        number (float): number to check

    Returns:
        nothing.

    Raises:
        ValueError: if number is negative
    """
    ''' '''
    if number < 0:
        logger.error('{0} is negative'.format(number))
        raise ValueError('{0} should not be negative'.format(number))
    return None
