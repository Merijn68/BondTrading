import pandas as pd
import numpy as np

import sys
from pathlib import Path
from typing import Tuple, Union
from loguru import logger

def add_duration(
  df: pd.DataFrame
):
  logger.info('Add remaining duration...')
  df['remain_duration'] = df['mature_dt'] - df['rate_dt'] 
    
  return df

def add_term_spread(
  df: pd.DataFrame
):
  logger.info('Add term spread...')
  mid_10y = df['y_bid10'] + df['y_offer10'] / 2
  mid_2y= df['y_bid2'] + df['y_offer2'] / 2
    
  df['term_spread'] = mid_10y - mid_2y

  return df

def add_bid_offer_spread(
  df: pd.DataFrame
) -> pd.DataFrame:
  logger.info('Add bid offer spread...')
  years =  [1,2,3,4,5,6,7,8,9,10,15,20,30]  
  bids = [''.join(('y_bid',str(year))) for year in years]
  offers =  [''.join(('y_offer',str(year))) for year in years]
  df_spread = pd.DataFrame(df[bids].to_numpy() - -df[offers].to_numpy())
  columns = [''.join(('y_spread_',str(year))) for year in years]
  df_spread.columns = columns

  df = pd.concat([df,df_spread], axis = 1 )

  return df



def country_spread(
  df  
) -> pd.DataFrame:
  import itertools

  countrydict: dict = {'DE': 'Germany','FR': 'France','ES': 'Spain','IT': 'Italy','US': 'United States','NL': 'Netherlands'}

  logger.info('Add Country Spread')
  countries = df['country'].unique()
  variables = ['bid','offer']
  df_pivot = pd.pivot(df, index = ['rate_dt','timeband','actual_dt', 'time'], columns = ['country'], values = variables)
  for variable in variables:    
    for combi in itertools.combinations(countries,2):            
        c0 = list(countrydict.keys())[list(countrydict.values()).index(combi[0])]
        c1 = list(countrydict.keys())[list(countrydict.values()).index(combi[1])]
        df_pivot['cs_'+c0+c1] = df_pivot[variable,combi[0]] - df_pivot[variable,combi[1]]

  df_pivot = df_pivot.drop(['bid','offer'], axis=1, level=0)
  df_pivot.columns = df_pivot.columns.droplevel(1)
  df_pivot = df_pivot.reset_index()
  
  return df_pivot




