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

def add_term_spread(
  df: pd.DataFrame
):

    mid_10y = df['yield_bid10'] + df['yield_offer10'] / 2
    mid_1y= df['yield_bid1'] + df['yield_offer1'] / 2
    
    df['term_spread'] = mid_10y - mid_1y

    return df

def add_bidoffer_spread(
  df: pd.DataFrame
) -> pd.DataFrame:
  years =  [1,2,3,4,5,6,7,8,9,10,15,20,30]  
  bids = [''.join(('yield_bid',str(year))) for year in years]
  offers =  [''.join(('yield_offer',str(year))) for year in years]
  df_spread = pd.DataFrame(df[bids].to_numpy() - -df[offers].to_numpy())
  columns = [''.join(('yield_spread_',str(year))) for year in years]
  df_spread.columns = columns
  return df



