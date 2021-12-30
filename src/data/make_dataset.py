import pandas as pd
import numpy as np
#from pandas.core.tools.datetimes import Datetime

import sys
from pathlib import Path
from typing import Tuple, Union
from loguru import logger



sys.path.insert(0, "..")




def read_csv(
    path: Path = Path("../data/raw/bonds.csv"),
    thousands:str = ','    
) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path, thousands=thousands )

    except Exception as error:      
        logger.error(f"Error loading data: {error}")

    return df


def get_bond_data(        
    ccy: str = '',
    path: Path = Path("../data/raw/bonds.csv"),
) -> pd.DataFrame:
    logger.info('Load bond data')
    ''' Read raw bond data from CSV, drop columns and format data '''    
    
    # loading bond data        
    df = read_csv(path)
    
    # Drop unneeded columns
    df = df.drop(['bondname','bond_ext_name', 'group_name','fix_float','cparty_type','CO2_factor'], axis = 1)

    # Correct columns types
    for column in ['cfi_code','isin','ccy','issuer_name','coupon_frq','country_name','issue_rating']:
        df[column] = df[column].astype('string')

    for column in ['issue_dt','first_coupon_date','mature_dt' ]:
        df[column] = pd.to_datetime(df[column])
       
    # Remove trailing spaces
    df['isin'] = df['isin'].str.strip()
    df['country_name'] = df['country_name'].str.strip()    
    
    # Total Issue amount '-' should be converted to 0
    df = df.rename(columns={' tot_issue ': 'tot_issue'})  
    df.loc[df['tot_issue'] == '-', 'tot_issue'] = '0'  
    df['tot_issue']= df['tot_issue'].str.replace(',','').str.replace('-','0').astype('float')    
    
    return df


# Subset data on country name and currency
def subset_bonds(
        df: pd.DataFrame,
        country: dict,
        ccy: str = 'EUR'         
) -> pd.DataFrame:
    
    df = df[df['country_name'].isin(country_dict.values())]
    # Lets select only EUR bonds - we don't want to complicate things with FX volatility
           
    if ccy:
        df = df[df['ccy'] == ccy]

    return df


def impute_bonds(
     df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute bond data')
    # FillNa Issue Rating missing for US and Wallonie    
    selection = ( df['issuer_name'] == 'UNITED STATES TREASURY' )
    df.loc[selection,'issue_rating'] = df.loc[selection,'issue_rating'].fillna('AAA')
    selection = ( df['issuer_name'] == 'WALLONIE' )
    df.loc[selection,'issue_rating'] = df.loc[selection,'issue_rating'].fillna('AA-')

    # CFI Codes Imputed with Dept Instrument XXXXX (Unknown)
    df['cfi_code'] = df['cfi_code'].fillna('DXXXXX')

    # Date 1899-12-30 means no first coupon date was known...
    df.loc[df['first_coupon_date'] == '30-12-1899', 'first_coupon_date'] = None    
    df.loc[df['issue_dt'] == '30-12-1899', 'issue_dt'] = None    
    df.loc[df['mature_dt'] == '30-12-1899', 'mature_dt'] = None    

    # impute first coupon date if missing
    df['first_coupon_date'] = df['first_coupon_date'].fillna(df['issue_dt'])
    df['bond_duration'] = df['mature_dt'] - df['first_coupon_date']    

    return df
    
def save_bonds(
    df: pd.DataFrame,    
) -> pd.DataFrame:
    # Store processed data
    logger.info('Save preprocessed bond data')
    try:
        df.to_csv('../data/processed/bonds.csv')
    except Exception as error:      
        logger.error(f"Error saving data: {error}")

# For sampling business Days
#    from pandas.tseries.offsets import BDay
#    pd.date_range('2015-07-01', periods=5, freq=BDay())

def get_price(
    ids: np.array  = [],
    path: Path = Path("../data/raw/price.csv"),
) -> pd.DataFrame:
    '''
        Load price data
        If array of ISIN's is given only prices for these ISINs will be returned.
        But all prices are loaded first...

    '''
    logger.info('Load bond price data')
    
    df = read_csv(path)

    # Correct columns types
    for column in ['reference_identifier','ccy']:
        df[column] = df[column].astype('string')
    for column in ['rate_dt' ]:
        df[column] = pd.to_datetime(df[column])

    # Filter on referenceId
    #if ids:
    #    df = df[df['reference_identifier'].isin(ids)]

    # Bid en Offer zijn overbodig want gelijk in bron systeem.
    # Deze voegen we samen
    df['mid'] = (df['bid'] + df['offer'])  / 2
    df = df.drop(['bid','offer'], axis = 1)    
    
    return df

def impute_price(
     df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute bond price')
    # 99.999 is een 'default'. Deze verwijderen we
    df = df[df['mid'] != 99.999]    
    return df


def get_yield_curves(
    path: Path = Path("../data/raw/yield.csv"),
) -> pd.DataFrame:
    logger.info('Load goverment yield curve data')

    country_dict = {'DE': 'Germany','FR': 'France','ES': 'Spain','IT': 'Italy','NL': 'Netherlands'}

    df = read_csv(path)

    # Voeg een country_name kolom toe
    for country in country_dict:
        df.loc[df['name'].str.contains(country), 'country'] = country_dict[country]

    # hernoem name naar ratename (voor de duidelijkheid)
    df = df.rename(columns={'name': 'ratename'})

    for column in ['ratename','ccy','timeband','int_basis','country']:
        df[column] = df[column].astype('string')
    for column in ['rate_dt','actual_dt' ]:
        df[column] = pd.to_datetime(df[column])
    
    return df    

def impute_yield_curves(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute yield curve')
    
    # Drop MM curves only Bond Based curves are actually correct
    df = df[df['ratename'].str.contains('BB')]    
    
    return df

def save_yield_curves(
    df: pd.DataFrame,    
):
    logger.info('Save preprocessed yield curve data')
    try:
        df.to_csv('../data/processed/yield.csv')
    except Exception as error:      
        logger.error(f"Error saving data: {error}")

    


def make_data(
):
    ''' Generate all data preprocessing '''
    df_bonds = get_bond_data()
    df_bonds = impute_bonds(df_bonds)
    save_bonds(df_bonds)




def get_data(    
    ids: np.array  = [],    
) -> pd.DataFrame:
    ''' 
        Get a join of bond en price data for selected bonds - for further analyses
    '''
    # Load all EUR bonds
    df_bonds = get_bond_data(ccy = 'EUR')

    df_bonds = df_bonds.drop('ccy', axis = 1)
    
    # Get bond prices of the selected bonds
    df_price = get_price(df_bonds.index.to_list())

    df = df_price.merge(df_bonds, left_on = 'reference_identifier', right_index=True,  how = 'left')
    df['reference_identifier'] = df['reference_identifier'].astype('string')
    return df

