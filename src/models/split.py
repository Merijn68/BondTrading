
import pandas as pd
from loguru import logger
from typing import Tuple


def split_data(
    df          :  pd.DataFrame,
    train_perc  :  float = 0.70,
    val_perc    :  float = 0.20,
    test_perc   :  float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    n = len(df)
    df_train = df[0:int(n*train_perc)]
    df_val = df[int(n*train_perc):int(n*(train_perc+val_perc))]
    df_test = df[int(n*(1-test_perc)):]

    logger.info(f'Data {len(df)}')
    logger.info(f'Training {len(df_train)}')
    logger.info(f'Validation {len(df_val)}')
    logger.info(f'Test {len(df_test)}')

    return (df_train, df_val, df_test)