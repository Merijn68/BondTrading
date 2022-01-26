"""
    Pre process data from raw to processed data


"""


import pandas as pd
import numpy as np
#from pandas.core.tools.datetimes import Datetime

import sys
from pathlib import Path
from typing import Tuple, Union, List
from loguru import logger
from src.features import build_features
from datetime import date

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
        df = pd.read_csv(path, thousands=thousands, dayfirst=True, *args, **kwargs )

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
    df = read_csv(path, parse_dates = ['issue_dt', 'first_coupon_date', 'mature_dt'])
    # remove special character
    df.columns = df.columns.str.replace(' ', '')
    
    # Drop unneeded columns
    df = df.drop(['bondname', 'group_name','fix_float','cparty_type','CO2_factor'], axis = 1)

    # Correct columns types
    for column in ['cfi_code','isin','ccy','issuer_name','coupon_frq','country_name','issue_rating']:
        df[column] = df[column].astype('string')

    #for column in ['issue_dt','first_coupon_date','mature_dt' ]:
    #    df[column] = pd.to_datetime(df[column])
       
    # Remove trailing spaces
    for col in ['isin','coupon_frq','country_name','issuer_name']:
        df[col] = df[col].str.strip()
        
    
    # rename country_name to country
    df = df.rename(columns={'country_name':'country'})  
    
    # Total Issue amount '-' should be converted to 0
    df.loc[df['tot_issue'] == '-', 'tot_issue'] = '0'  
    df['tot_issue']= df['tot_issue'].str.replace(',','').str.replace('-','0').astype('float')  / 1000000  

    df['coupon'] = df['coupon'].str.replace(',','').str.replace('-','0').astype('float')

    return df


def impute_bonds(
     df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info('Impute bond data')
    
    # CFI Codes Imputed with Dept Instrument XXXXX (Unknown)
    df['cfi_code'] = df['cfi_code'].fillna('DXXXXX')

    # Date 1899-12-30 means no first coupon date was known...
    df.loc[df['first_coupon_date'] == '30-12-1899', 'first_coupon_date'] = None    
    df.loc[df['issue_dt'] == '30-12-1899', 'issue_dt'] = None    
    df.loc[df['mature_dt'] == '30-12-1899', 'mature_dt'] = None    

    # impute first coupon date if missing
    df['first_coupon_date'] = df['first_coupon_date'].fillna(df['issue_dt'])

    # Bond duration (estimation in dagen)
    df['bond_duration'] = df['mature_dt'] - df['issue_dt']    
    df['bond_duration'] = df['bond_duration'].dt.days   


    # Fill missing issuer_rating met meest voorkomende waarde per issuer name
    # voor nu verwijderen we deze bonds
    df = df[~df['issue_rating'].isnull()]    

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
    
    df = read_csv(path,  parse_dates = ['rate_dt'])
    # remove special character
    df.columns = df.columns.str.replace(' ', '')
    
    # Correct columns types
    for column in ['reference_identifier','ccy']:
        df[column] = df[column].astype('string')
   
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

    # data van 31-12 is een duplicaat van 30-12. Deze verwijderen    
    from datetime import date
    df['lastday'] = df['rate_dt'].dt.year.apply( lambda x : date( x, 12, 31))
    df = df[df['rate_dt'] != df['lastday']].copy()
    df.drop('lastday', axis = 'columns')
    # 99.999 is een 'default'. Deze verwijderen we
    df = df[df['mid'] != 99.999]    

    
    return df

def get_yield(
    path: Path = Path("../data/raw/yield.csv"),
) -> pd.DataFrame:
    logger.info('Load goverment yield curve data')

    country_dict = {'DE': 'Germany','FR': 'France','ES': 'Spain','IT': 'Italy','NL': 'Netherlands'}
    
    df = read_csv(path, parse_dates = ['rate_dt', 'actual_dt'])
    # remove special character
    df.columns = df.columns.str.replace(' ', '')
    
    # Voeg een country_name kolom toe
    for country in country_dict:
        df.loc[df['ratename'].str.contains(country), 'country'] = country_dict[country]
    
    # Data bevat meerdere waarnemingen per dag. Bewaar alleen de laatste
    df = df.groupby(['country','rate_dt','timeband']).nth(-1)
    df = df.reset_index()

    for column in ['ratename','ccy','timeband','int_basis','country']:
        df[column] = df[column].astype('string')    
        df[column] = df[column].str.strip()

    for column in ['bid','offer']:
        df[column] = df[column].replace(',','').replace('-','0').astype('float')            
    
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

    # Bereken alvast de mid rate
    df['mid'] = (df['bid'] + df['offer'] ) / 2

    # drop data voor 1-jan-2010
    df[df['rate_dt'] >= '1-jan-2010']
    
    return df


    
def get_inflation(    
    countrydict: dict = {'DE': 'Germany','FR': 'France','ES': 'Spain','IT': 'Italy','US': 'United States'},
    path: Path = Path("../data/raw"),    
)    -> pd.DataFrame:
    logger.info('Load goverment yield curve data')
    
    df = pd.DataFrame()
    for country in countrydict:
        filepath = Path(path, f"{country} Inflation.csv")
        dfc = read_csv(filepath, skiprows = 2,  header=None)
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
    df_bp = build_features.add_duration(df_bp)    
    save_pkl('bp', df_bp)

    df_tf = build_simple_input(df_bonds, df_price)    
    save_pkl('tf', df_tf)

    df_bpy = join_yield(df_bp, df_yield)    
    df_bpy = build_features.add_term_spread(df_bpy)
    df_bpy = build_features.add_bid_offer_spread(df_bpy)
    save_pkl('bpy', df_bpy)


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

def join_infl(
    df_bpy: pd.DataFrame,    
    df_inflation: pd.DataFrame,        
    country: str = 'Germany'
) -> pd.DataFrame:    

    if country == '':
        df_inflation = pd.pivot(df_inflation, index = ['country','rate_dt'], columns = ['timeband'], values = 'inflation')
        df = df_bpy.merge(df_inflation, left_on = ['country','rate_dt'], right_index=True, how = 'inner')
    else:
        df_inflation = df_inflation[df_inflation['country'] == country]
        df_inflation = pd.pivot(df_inflation, index = ['rate_dt'], columns = ['timeband'], values = 'inflation')
        df = df_bpy.merge(df_inflation, left_on = ['rate_dt'], right_index=True, how = 'inner')
        
    df = df.dropna(subset = ['10 YEARS'])

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


def join_ytm(
    df_bp: pd.DataFrame,
    df_yield: pd.DataFrame
) -> pd.DataFrame:
    ''' Calculate for each bond per rate datae the Yield to Maturity '''
    df_y = df_yield[['country', 'rate_dt','time', 'mid']].rename(columns={ 'mid': 'ytm'})
    df = df_bp.merge(df_y, on = ['country','rate_dt'], how = 'inner')
    df = df[ df['time'].dt.days > df.remain_duration ]
    df = df.sort_values(by = ['reference_identifier','rate_dt','time'])
    df = df.groupby(['reference_identifier','rate_dt']).first()
    df = df.reset_index()

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
        df.to_pickle(f'../data/processed/{name}.pkl', protocol = 4)  # Set protocol to 4 for Colab
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

def build_simple_input(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame
) -> pd.DataFrame:
    ''' Build a simple format for tensorflow '''

    df_bp = join_price(df_bonds,df_price )    
    df_bp = build_features.add_duration(df_bp)    

    # Alle geselecteerde bonds zijn in EUR. Referrence_identifier en ISIN zijn dubbel
    df_bp = df_bp.drop(['ccy','reference_identifier'], axis = 1)    
    df_bp = build_features.encode_coupon_freq(df_bp)    
    df_bp = build_features.encode_cfi(df_bp)

    return df_bp


def split_data(
    df          :  pd.DataFrame,
    train_perc  :  float = 0.70,
    val_perc    :  float = 0.20,
    test_perc   :  float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    n = len(df)
    df_train = df[0:int(n*train_perc)]
    df_val = df[int(n*train_perc):int(n*(train_perc+val_perc))]
    df_test = df[int(n*(1-test_perc)):]

    logger.info(f'Data {len(df)}')
    logger.info(f'Training {len(df_train)}')
    logger.info(f'Validation {len(df_val)}')
    logger.info(f'Test {len(df_test)}')

    return (df_train, df_val, df_test)

def read_single_bond(
    isin:   str,
    train_perc  :  float = 0.70,
    val_perc    :  float = 0.20,
    test_perc   :  float = 0.10,
) -> pd.DataFrame:

    df = read_pkl('price')
    df = df[df['reference_identifier'] == isin]
    df_train, df_val, df_test = split_data(df, train_perc, val_perc, test_perc)

    df_train = df_train.drop(['ccy','reference_identifier','rate_dt'], axis = 'columns')
    df_val = df_val.drop(['ccy','reference_identifier','rate_dt'], axis = 'columns')
    df_test = df_test.drop(['ccy','reference_identifier','rate_dt'], axis = 'columns')
    
    return (df_train, df_val, df_test)

def read_bond_with_features( 
    isin        :   str,
    features    :   List[str] = [],
    train_perc  :   float = 0.70,
    val_perc    :   float = 0.20,
    test_perc   :   float = 0.10,
    label_var   :   str = 'mid',
    time_var    :   str = 'rate_dt'
) -> pd.DataFrame:

    df_price = read_pkl('price')    
    df_bonds = read_pkl('bonds')   

    # Select a single bond 
    df_bonds = df_bonds[df_bonds['isin'] == isin]
    df_price = df_price[df_price['reference_identifier'] == isin]

    df = build_simple_input(df_bonds, df_price )

    if features:
        variables = [label_var, *features] 
        df = df.filter(variables)

    df_train, df_val, df_test = split_data(df, train_perc, val_perc, test_perc)


    return (df_train, df_val, df_test)


