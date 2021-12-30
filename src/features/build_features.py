import pandas as pd
import numpy as np

import sys
from pathlib import Path
from typing import Tuple, Union
from loguru import logger

def add_duration(
  df: pd.DataFrame
):

    df['remain_duration'] = df['mature_dt'] - df['rate_dt'] 
    
    return df

