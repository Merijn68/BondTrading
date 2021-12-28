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
    path: Path = Path("../data/raw/bonds.csv"),
) -> pd.DataFrame:
    logger.info('Load bond data')
    '''
        69 Governement bonds. Bondname, bond_ext_name en isin zijn allemaal sleutelwaarden.
        group_name is overbodig (allemaal government bonds).
        fix_float is overbodig (allemaal fixed rate bonds).
        cparty_type is overbodig (allemaal government bonds)
        tot_issue columns seems to have a space in the name?

        first_coupon_date, mature_date -> datum velden
        issuer name is Categorical
        coupon_frq is Categorical

        Inlezen bond definities en omzetten naar categorische of float variabelen

        Berekenen:
            CFI Code splitsen per letter
            Oorsponkelijke looptijd berekenen (maturity date - issue date)
    '''
    # loading bond data    
    df = read_csv(path)

    df.drop(['bondname','bond_ext_name', 'group_name','fix_float','cparty_type','CO2_factor'], axis = 1, inplace = True)
    
    df['issue_dt']= pd.to_datetime(df['issue_dt'])
    df['first_coupon_date'] = pd.to_datetime(df['first_coupon_date'])
    df['mature_dt']= pd.to_datetime(df['mature_dt'])

    df = df.rename(columns={' tot_issue ': 'tot_issue'})    
    df = df.set_index('isin')

    df['issuer_name'] = df['issuer_name'].astype('category')
    df['coupon_frq'] = df['issuer_name'].astype('category')
    df['issue_rating'] = df['issue_rating'].astype('category')  

    df['cfi_code1'] = df['cfi_code'].str[:1].astype('category')
    df['cfi_code2'] = df['cfi_code'].str[1:2].astype('category')
    df['cfi_code3'] = df['cfi_code'].str[2:3].astype('category')
    df['cfi_code4'] = df['cfi_code'].str[3:4].astype('category')
    df['cfi_code5'] = df['cfi_code'].str[4:5].astype('category')
    df['cfi_code6'] = df['cfi_code'].str[5:6].astype('category')    
    df = df.drop(['cfi_code'], axis = 1)

    # impute first coupon date if missing
    df['first_coupon_date'] = df['first_coupon_date'].fillna(df['issue_dt'])
    df['bond_duration'] = df['mature_dt'] - df['first_coupon_date']    
    
    return df

# For sampling business Days
#    from pandas.tseries.offsets import BDay
#    pd.date_range('2015-07-01', periods=5, freq=BDay())

def get_price(
    path: Path = Path("../data/raw/price.csv"),
) -> pd.DataFrame:
    logger.info('Load bond price data')
    
    df = read_csv(path)
    df['reference_identifier'] = df['reference_identifier'].astype('string')
    df['ccy'] = df['ccy'].astype('string')
    df['rate_dt'] =  pd.to_datetime(df['rate_dt'])
    df['mid'] = (df['bid'] + df['offer'])  / 2
    df = df.drop(['bid','offer'], axis = 1)    
    return df