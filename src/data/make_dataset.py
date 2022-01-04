"""
    Pre process data from raw to processed data

"""


import pandas as pd
import numpy as np
#from pandas.core.tools.datetimes import Datetime

import sys
from pathlib import Path
from typing import Tuple, Union
from loguru import logger
from src.features import build_features

sys.path.insert(0, "..")


def read_csv(
    path: Path = Path("../data/raw/bonds.csv"),
    thousands:str = ',',
    *args,
    **kwargs    
) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path, thousands=thousands, *args, **kwargs )

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
    for col in ['isin','coupon_frq','country_name']:
        df[col] = df[col].str.strip()
        
    # Total Issue amount '-' should be converted to 0
    # rename country_name to country
    df = df.rename(columns={' tot_issue ': 'tot_issue', 'country_name':'country'})  

    # Save tot_issue as amount
    df.loc[df['tot_issue'] == '-', 'tot_issue'] = '0'  
    df['tot_issue']= df['tot_issue'].str.replace(',','').str.replace('-','0').astype('float')    
    
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
    # Bond duration (estimation)
    df['bond_duration'] = df['mature_dt'] - df['issue_dt']    

    return df
 

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

    # Bid en Offer zijn overbodig want gelijk in bron systeem.
    # Deze voegen we samen
    df['mid'] = (df['bid'] + df['offer'])  / 2
    df = df.drop(['bid','offer'], axis = 1)    

    # Data van 30-7 zit 2x in de bron
    df = df.drop_duplicates()

    
    return df

def impute_price(
     df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute bond price')
    # 99.999 is een 'default'. Deze verwijderen we
    df = df[df['mid'] != 99.999]    
    return df

def get_yield(
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
    
    df['timeband'] = df['timeband'].str.strip()

    # Data bevat meerdere waarnemingen per dag. Bewaar alleen de laatste
    df = df.groupby(['country','rate_dt','timeband']).nth(-1)
    df = df.reset_index()

    return df    


def impute_yield(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute yield curve')

    # Drop MM curves only Bond Based curves are actually correct
    df = df[df['ratename'].str.contains('BB')].copy()    

    # Herberekenen actual dt -> zodat deze in lijn is met de inflatie data (geen rekening houden met holidays)     
    df['offset'] = df['timeband'].str.extract('(\d+)')
    df['actual_dt'] = df['rate_dt'] + df['offset'].astype('timedelta64[Y]')

    # De timeband omzetten naar een timedelta.
    df['time'] = df['actual_dt'] - df['rate_dt']
    df = df.drop(columns = 'offset')
    
    return df


    
def get_inflation(    
    countrydict: dict = {'DE': 'Germany','FR': 'France','ES': 'Spain','IT': 'Italy','US': 'United States'}
)    -> pd.DataFrame:
    logger.info('Load goverment yield curve data')
    
    df = pd.DataFrame()
    for country in countrydict:
        path = Path(f"../data/raw/{country} Inflation.csv")
        dfc = read_csv(path, skiprows = 2,  header=None)
        if not dfc.empty:
            dfc = dfc.iloc[:,2:17]        
            dfc.columns = ['rate_dt','1 YEAR','2 YEARS', '3 YEARS','4 YEARS', '5 YEARS','6 YEARS','7 YEARS','8 YEARS','9 YEARS','10 YEARS',
            '15 YEARS','20 YEARS','25 YEARS','30 YEARS']
            dfc['country'] = countrydict[country]

            df = pd.concat([df,dfc])

    df = df.melt(id_vars= ['country', 'rate_dt'],var_name = 'timeband', value_name = 'inflation')
    df['ratename'] = 'Inflation'  

    for column in ['ratename','timeband','country']:
        df[column] = df[column].astype('string')
    for column in ['rate_dt']:
        df[column] = pd.to_datetime(df[column])
    return df

def impute_inflation(
    df: pd.DataFrame,
) -> pd.DataFrame:


    logger.info('Impute inflation curve')

    # Drop all NANs    
    df = df[~df['inflation'].isnull()].copy()

    # Translate timeband to actual_dt
    df['offset'] = df['timeband'].str.extract('(\d+)')
    df['actual_dt'] = df['rate_dt'] + df['offset'].astype('timedelta64[Y]')
    
    # De timeband omzetten naar een timedelta.
    df['time'] = df['actual_dt'] - df['rate_dt']
    
    df = df.drop(columns = 'offset')

    return df

def make_data(
):
    ''' Generate all data preprocessing '''
    df_bonds = get_bond_data()
    df_bonds = impute_bonds(df_bonds)
    save_pkl('bonds', df_bonds)

    df_price = get_price()
    df_price = impute_price(df_price)
    save_pkl('price', df_price)

    df_yield = get_yield()
    df_yield = impute_yield(df_yield)    
    save_pkl('yield', df_yield)

    df_inflation = get_inflation()
    df_inflation = impute_inflation(df_inflation)    
    save_pkl('inflation', df_inflation)
    
    df_bp = join_price(df_bonds,df_price )
    save_pkl('bp', df_bp)

    df_tf = build_simple_input(df_bonds, df_price)    
    save_pkl('tf', df_tf)

    df_bpy = join_yield(df_bp, df_yield)    
    df_bpy = build_features.add_term_spread(df_bpy)
    df_bpy = build_features.add_bid_offer_spread(df_bpy)
    save_pkl('bpy', df_bpy)


def build_simple_input(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame
) -> pd.DataFrame:
    ''' Build a first simple format for tensorflow '''

    df_bp = join_price(df_bonds,df_price )    
    df_bp = build_features.add_duration(df_bp)    

    df_bp['bond_duration'] = df_bp['bond_duration'].dt.days
    df_bp['remain_duration'] = df_bp['remain_duration'].dt.days

    # Alle geselecteerde bonds zijn in EUR. Referrence_identifier en ISIN zijn dubbel
    df_bp = df_bp.drop(['ccy','reference_identifier'], axis = 1)    
    df_bp = build_features.encode_coupon_freq(df_bp)    
    df_bp = build_features.encode_cfi(df_bp)

    # Alle string variabelen worden one hot encoded...
    for column in df_bp.select_dtypes(include=['string']).columns.tolist():
        df_bp = build_features.encode_onehot(df_bp,column)
    
    return df_bp

    
def join_price(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame
) -> pd.DataFrame:
    
    df_bonds = df_bonds.drop('ccy', axis = 1)                          
    df = df_price.merge(df_bonds, left_on = 'reference_identifier', right_on='isin',  how = 'left')
        
    return df

def join_inflation(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame,
    df_inflation: pd.DataFrame,        
) -> pd.DataFrame:
        
    df = join_price(df_bonds, df_price)

    df_inflation = pd.pivot(df_inflation, index = ['country','rate_dt'], columns = ['timeband'], values = 'inflation')
    df = df.merge(df_inflation, left_on = ['country','rate_dt'], right_index=True, how = 'inner')

    return df    

def join_yield(    
    df: pd.DataFrame,
    df_yield: pd.DataFrame,    
) -> pd.DataFrame:
    
    df_yield['offset'] = df_yield['timeband'].str.extract('(\d+)')

    # Join yield
    df_yield_pivot = pd.pivot(df_yield, index = ['country','rate_dt'], columns = ['offset'], values = ['bid','offer'])
    df_yield_pivot.columns = [''.join(('y_',''.join(tup))).strip() for tup in df_yield_pivot.columns]
    columns = df_yield_pivot.columns.to_list()
    columns.sort(key=lambda x: int(x.replace('y_bid','').replace('y_offer','')))
    df_yield_pivot = df_yield_pivot[columns]
    df = df.merge(df_yield_pivot, left_on = ['country','rate_dt'], right_index=True, how = 'inner')

    
    return df    

def fulljoin(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame,
    df_inflation: pd.DataFrame,    
    df_yield: pd.DataFrame,        
    df_countryspread: pd.DataFrame,  
) -> pd.DataFrame:

    # Join bond price
    df = join_price(df_bonds, df_price)


    # Join inflation
    df_inflation_pivot = pd.pivot(df_inflation, index = ['country','rate_dt'], columns = ['timeband'], values = 'inflation')    
    df_inflation_pivot.columns = [''.join(('inflation_',col)).replace('YEARS','').replace('YEAR','').strip() for col in df_inflation_pivot.columns]
    columns = df_inflation_pivot.columns.to_list()
    columns.sort(key=lambda x: int(x[10:]))
    df_inflation_pivot = df_inflation_pivot[columns]

    df = df.merge(df_inflation_pivot, left_on = ['country','rate_dt'], right_index=True, how = 'inner')

    # Join yield
    df_yield_pivot = pd.pivot(df_yield, index = ['country','rate_dt'], columns = ['timeband'], values = ['bid','offer'])
    df_yield_pivot.columns = [''.join(('yield_',''.join(tup))).replace('YEARS','').replace('YEAR','').strip() for tup in df_yield_pivot.columns]
    columns = df_yield_pivot.columns.to_list()
    columns.sort(key=lambda x: int(x[6:].replace('bid','').replace('offer','')))
    df_yield_pivot = df_yield_pivot[columns]

    df = df.merge(df_yield_pivot, left_on = ['country','rate_dt'], right_index=True, how = 'inner')

    # Join country spread    
    df_countryspread = pd.pivot(df_countryspread, index = ['rate_dt'], columns = ['timeband'])
        

    return df    


def save_pkl(
    name: str,
    df: pd.DataFrame 
):
    # Store processed data
    logger.info(f'Save preprocessed {name} data')
    try:
        df.to_pickle(f'../data/processed/{name}.pkl')
        df.to_csv(f'../data/processed/{name}.csv')
    except Exception as error:      
        logger.error(f"Error saving {name} data: {error}")

def read_pkl(
    name: str,
    path = Path("../data/processed/"),    
    filename_suffix = 'pkl'
) -> pd.DataFrame:
    # load processed data
    logger.info(f'Load preprocessed {name} data')
    df = pd.DataFrame()
    if not path.is_dir():
        logger.error(f'Directory {path} not found')
    else:
        path = Path(path, name + "." + filename_suffix)
        if not path.exists():
            logger.error(f'File {path} not found')
        else:                
            df = pd.read_pickle(path)
    
    return df


def read_single_bond(
    isin: str
) -> pd.DataFrame:

    df = read_pkl('bp')
    df = df[df['reference_identifier'] == isin].copy()

    return df