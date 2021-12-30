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
    df = df.drop(['bondname','bond                                                                                                                                                                    _ext_name', 'group_name','fix_float','cparty_type','CO2_factor'], axis = 1)

    # Correct columns types
    for column in ['cfi_code','isin','ccy','issuer_name','coupon_frq','country_name','issue_rating']:
        df[column] = df[column].astype('string')

    for column in ['issue_dt','first_coupon_date','mature_dt' ]:
        df[column] = pd.to_datetime(df[column])
       
    # Remove trailing spaces
    df['isin'] = df['isin'].str.strip()
    df['country_name'] = df['country_name'].str.strip()    
    
    # Total Issue amount '-' should be converted to 0
    df = df.rename(columns={' tot_issue ': 'tot_issue', 'country_name':'country'})  

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
    
    return df    


def impute_yield(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute yield curve')

    # Drop MM curves only Bond Based curves are actually correct
    df = df[df['ratename'].str.contains('BB')]    
    
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
    df = df[~df['inflation'].isnull()]
    
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


def join_bond_data(    
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame,
    ids: np.array  = [],  
    ccy: str = 'EUR'  
) -> pd.DataFrame:
    ''' 
        Get a join of bond en price data for selected bonds - for further analyses
    '''
    # Load only 1 curremcy
    df_bonds = df_bonds[df_bonds['ccy'] == ccy]
    df_bonds = df_bonds[df_bonds['isin'].isin(ids)]
    df_bonds = df_bonds.drop('ccy', axis = 1)      
    
    df_price = df_price[df_price['ccy'] == ccy]
    df_price = df_price[df_price['reference_identifier'].isin(ids)]

    df = df_price.merge(df_bonds, left_on = 'reference_identifier', right_on='isin',  how = 'left')
    
    return df

def join_full(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame,
    df_yield: pd.DataFrame,
    df_inflation: pd.DataFrame
) -> pd.DataFrame:
    ids: np.array  = [],  
    ccy: str = 'EUR'  

    # Load only 1 curremcy
    df_bonds = df_bonds[df_bonds['ccy'] == ccy]
    df_bonds = df_bonds[df_bonds['isin'].isin(ids)]
    df_bonds = df_bonds.drop('ccy', axis = 1)      
    
    df_price = df_price[df_price['ccy'] == ccy]
    df_price = df_price[df_price['reference_identifier'].isin(ids)]

    df = df_price.merge(df_bonds, left_on = 'reference_identifier', right_on='isin',  how = 'left')
    
    df_inflation = pd.pivot(df_inflation, index = ['country','rate_dt'], columns = ['timeband'], values = 'inflation')

    df = df.merge(df_inflation, left_on = ['country','rate_dt'], right_index=True)
    return df
    

def save_pkl(
    name: str,
    df: pd.DataFrame 
):
    # Store processed data
    logger.info(f'Save preprocessed {name} data')
    try:
        df.to_pickle(f'../data/processed/{name}.pkl')
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
