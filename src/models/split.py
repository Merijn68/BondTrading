import pandas as pd
from loguru import logger
from typing import Tuple, List


def split_data(
    df: pd.DataFrame, train_perc: float = 0.70, columns: List = []
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if columns:
        df = df[columns]
    test_perc = 1 - train_perc
    n = len(df)
    df_train = df[0 : int(n * train_perc)]
    df_test = df[int(n * (1 - test_perc)) :]
    tpc = train_perc * 100

    logger.info(
        f"Train test split data {len(df)} {tpc}%,"
        + f"train data {len(df_train)}, test data {len(df_test)}."
    )

    return (df_train, df_test)
