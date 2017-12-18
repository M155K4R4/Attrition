import time
import numpy as np
import pandas as pd

import config
import work_data
import models

def main():
    logger = config.ConfigLogger(__name__, 10)
    t0 = time.ctime()
    logger.info('OA')

    config.time_taken_display(t0)



if __name__=='__main__':
    main()
