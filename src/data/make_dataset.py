from datetime import date
import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalarOrArrayConvertible
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Union

from loguru import logger
from sklearn.model_selection import train_test_split


sys.path.insert(0, "..")

country_dict = {'DE': 'Germany','FR': 'France','ES': 'Spain','IT': 'Italy','NL': 'Netherlands'}


def read_csv(
    path: Path = Path("../data/raw/bonds.csv"),
) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path, thousands=',' )

    except Exception as error:      
        logger.error(f"Error loading data: {error}")

    return df


def get_bond_data(        
    ccy: str = '',
    path: Path = Path("../data/raw/bonds.csv"),
) -> pd.DataFrame:
    logger.info('Load bond data')
    '''        
        Inlezen bond definities, imputeren lege waarden en omzetten naar categorische of float variabelen

        Berekenen:
            CFI Code splitsen per letter
            Oorsponkelijke looptijd berekenen (maturity date - issue date)
    '''
    # loading bond data    
    df = read_csv(path)

    # Subset data on country name
    df['country_name'] = df['country_name'].str.strip()
    df = df[df['country_name'].isin(country_dict.values())]


    # Lets select only EUR bonds - we don't want to complicate things with FX volatility
    df['ccy'] = df['ccy'].astype('string')
    
    if ccy:
        df = df[df['ccy'] == ccy]

    df.drop(['bondname','bond_ext_name', 'group_name','fix_float','cparty_type','CO2_factor'], axis = 1, inplace = True)
    
    df['issue_dt']= pd.to_datetime(df['issue_dt'])
    df['first_coupon_date'] = pd.to_datetime(df['first_coupon_date'])
    df['mature_dt']= pd.to_datetime(df['mature_dt'])
    
    # Remove trailing spaces
    df['isin'] = df['isin'].str.strip()
    

    df = df.set_index('isin')
    
    # Total Issue amount '-' should be converted to 0
    df = df.rename(columns={' tot_issue ': 'tot_issue'})  
    df.loc[df['tot_issue'] == '-', 'tot_issue'] = '0'  
    df['tot_issue']= df['tot_issue'].str.replace(',','').str.replace('-','0').astype('float')    

    df['issuer_name'] = df['issuer_name'].astype('category')
    df['coupon_frq'] = df['coupon_frq'].astype('category')

    df['country_name'] = df['country_name'].astype('category')

    # FillNa Issue Rating missing for US and Wallonie    
    selection = ( df['issuer_name'] == 'UNITED STATES TREASURY' )
    df.loc[selection,'issue_rating'] = df.loc[selection,'issue_rating'].fillna('AAA')
    selection = ( df['issuer_name'] == 'WALLONIE' )
    df.loc[selection,'issue_rating'] = df.loc[selection,'issue_rating'].fillna('AA-')
    df['issue_rating'] = df['issue_rating'].astype('category')  
    
    # CFI Codes Imputed with Dept Instrument XXXXX (Unknown)
    df['cfi_code'] = df['cfi_code'].fillna('DXXXXX')

    df['cfi_code1'] = df['cfi_code'].str[:1].astype('category')
    df['cfi_code2'] = df['cfi_code'].str[1:2].astype('category')
    df['cfi_code3'] = df['cfi_code'].str[2:3].astype('category')
    df['cfi_code4'] = df['cfi_code'].str[3:4].astype('category')
    df['cfi_code5'] = df['cfi_code'].str[4:5].astype('category')
    df['cfi_code6'] = df['cfi_code'].str[5:6].astype('category')        

    # Date 1899-12-30 means no first coupon date was known...
    df.loc[df['first_coupon_date'] == '30-12-1899', 'first_coupon_date'] = None    
    df.loc[df['issue_dt'] == '30-12-1899', 'issue_dt'] = None    
    df.loc[df['mature_dt'] == '30-12-1899', 'mature_dt'] = None    

    # impute first coupon date if missing
    df['first_coupon_date'] = df['first_coupon_date'].fillna(df['issue_dt'])
    df['bond_duration'] = df['mature_dt'] - df['first_coupon_date']    

    # Store processed data
    df.to_csv('../data/processed/bonds.csv')

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
    df['reference_identifier'] = df['reference_identifier'].astype('string')

    # Filter on referenceId
    if ids:
        df = df[df['reference_identifier'].isin(ids)]

    df['ccy'] = df['ccy'].astype('string')
    df['rate_dt'] =  pd.to_datetime(df['rate_dt'])
    df['mid'] = (df['bid'] + df['offer'])  / 2
    df = df.drop(['bid','offer'], axis = 1)    

    # 99.999 is a 'default price'. Lets drop these.    
    df = df[df['mid'] != 99.999]    
    
    return df

def get_yield_curves(
    path: Path = Path("../data/raw/yield.csv"),
) -> pd.DataFrame:
    logger.info('Load goverment yield data')
    df = read_csv(path)
    df['name']= df['name'].astype('string')
    df['ccy'] = df['ccy'].astype('string')
    df['rate_dt'] =  pd.to_datetime(df['rate_dt'])
    df['timeband'] = df['timeband'].astype('string')
    df['actual_dt'] =  pd.to_datetime(df['actual_dt'])
    df['int_basis'] = df['int_basis'].astype('string')
    df = df.rename(columns={'name': 'ratename'})

    # Drop MM curves only Bond Based curves are actually correct
    df = df[df['ratename'].str.contains('BB')]    
    

    # Create a country_name column to match with bond data    
    for country in country_dict:
        df.loc[df['ratename'].str.contains(country), 'country'] = country_dict[country]
    df['country']= df['country'].astype('string')

    return df    








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

