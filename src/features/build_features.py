import pandas as pd
import numpy as np

import sys
from pathlib import Path
from typing import Tuple, Union
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

def add_duration(
  df: pd.DataFrame
) -> pd.DataFrame:
  logger.info('Add remaining duration...')
  df['remain_duration'] = df['mature_dt'] - df['rate_dt']   
  df['remain_duration'] = df['remain_duration'].dt.days    
    
  return df

def encode_coupon_freq(
  df: pd.DataFrame,
  col: str = 'coupon_frq'
) -> pd.DataFrame:
  logger.info(f'Encode coupon frequency {col}')
  # Tot nu toe 2 waarden: ANNUAL en SEMI ANUAL. Vertalen in Dagen
  # Encoding year as 364 to be 2x 182
  freq = { 'ANNUAL': 364, 'SEMI ANNUAL': 182}
  df[col].map(freq).fillna(0).astype(int)

  return df

def encode_onehot(
  df: pd.DataFrame,
  col: str
) -> pd.DataFrame:
  ''' We use sklearn module for one hot encoding '''
  logger.info(f'One Hot Encode column {col}')
  one_hot_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
  df_enc = pd.DataFrame(one_hot_encoder.fit_transform(df[[col]]),  columns=one_hot_encoder.categories_).add_prefix(col+'_')
  df_enc.columns = df_enc.columns.get_level_values(0)
  df = df.join(df_enc)

  df = df.drop(col, axis = 'columns')

  return df

def encode_cfi(
  df: pd.DataFrame,
  col: str = 'cfi_code'
) -> pd.DataFrame:

  logger.info(f'Encode CFI Code column {col}')

  # Split CFI in a column per character
  for x in range(6):
    df[col+'_'+str(x)] = df[col].str[x:x+1]
  
  # Drop the combined value
  df = df.drop(col, axis =  'columns')
  return df        

def add_term_spread(
  df: pd.DataFrame
) -> pd.DataFrame:
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

def add_estimated_bond_price(
 df: pd.DataFrame,
 var: str = 'estimated_bondprice'
) -> pd.DataFrame:
  ''' Calculate theoreritcal bond price based on yield to maturity'''
  df[var] = df.apply( calculate_price, axis = 1 )
  return df

def calculate_price( 
    row
) -> float:
         
   periods = (row['remain_duration'] / 365) 
   ytm = row['ytm'] / 100
   coupon = row['coupon'] / 100
   if (row['coupon_frq'] == 'ANNUAL'):
      frequency = 1
   else:
      frequency = 2
   par_value = 100    
   i = coupon / frequency * par_value
   p = (1 - ( 1 + ytm / frequency ) ** (-periods))/(ytm / frequency)
   pi = p * i
   k = par_value / ( ( 1 + ytm/frequency ) ** periods)
   price = k + pi
   return (price)

