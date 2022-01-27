""" 
    Handy routines to join datasets
"""

import pandas as pd

def join_price(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame
) -> pd.DataFrame:
    """ join bonds and price data on isin = reference_identifier """
    
    df_bonds = df_bonds.drop('ccy', axis = 1)                          
    df = df_price.merge(df_bonds, left_on = 'reference_identifier', right_on='isin',  how = 'left')
        
    return df


def yield_to_maturity(
    df_bp: pd.DataFrame,
    df_yield: pd.DataFrame
) -> pd.DataFrame:
    ''' Calculate for each bond per rate data the Yield to Maturity '''

    df_y = df_yield[['country', 'rate_dt','time', 'mid']].rename(columns={ 'mid': 'ytm'})
    df = df_bp.merge(df_y, on = ['country','rate_dt'], how = 'inner')
    df = df[ df['time'].dt.days > df.remain_duration ]
    df = df.sort_values(by = ['reference_identifier','rate_dt','time'])
    df = df.groupby(['reference_identifier','rate_dt']).first()
    df = df.reset_index()
    return df

def join_yield(    
    df: pd.DataFrame,
    df_yield: pd.DataFrame,    
) -> pd.DataFrame:
    """ join bond data with yield data on country and rate_dt """
    
    df_yield['offset'] = df_yield['timeband'].str.extract('(\d+)')

    # Join yield
    df_yield_pivot = pd.pivot(df_yield, index = ['country','rate_dt'], columns = ['offset'], values = ['bid','offer'])
    df_yield_pivot.columns = [''.join(('y_',''.join(tup))).strip() for tup in df_yield_pivot.columns]
    columns = df_yield_pivot.columns.to_list()
    columns.sort(key=lambda x: int(x.replace('y_bid','').replace('y_offer','')))
    df_yield_pivot = df_yield_pivot[columns]
    df = df.merge(df_yield_pivot, left_on = ['country','rate_dt'], right_index=True, how = 'inner')

    return df    

def join_inflation(
    df_bonds: pd.DataFrame,
    df_price: pd.DataFrame,
    df_inflation: pd.DataFrame,        
) -> pd.DataFrame:
    """ Join bonds, prices and inflation data """
        
    df = join_price(df_bonds, df_price)

    df_inflation = pd.pivot(df_inflation, index = ['country','rate_dt'], columns = ['timeband'], values = 'inflation')
    df = df.merge(df_inflation, left_on = ['country','rate_dt'], right_index=True, how = 'inner')

    return df    

def join_10y_inflation(
    df_bpy: pd.DataFrame,    
    df_inflation: pd.DataFrame,        
    country: str = 'Germany'
) -> pd.DataFrame:    
    """ add the 10 years inflation point for each bond price """

    if country == '':
        df_inflation = pd.pivot(df_inflation, index = ['country','rate_dt'], columns = ['timeband'], values = 'inflation')
        df = df_bpy.merge(df_inflation, left_on = ['country','rate_dt'], right_index=True, how = 'inner')
    else:
        df_inflation = df_inflation[df_inflation['country'] == country]
        df_inflation = pd.pivot(df_inflation, index = ['rate_dt'], columns = ['timeband'], values = 'inflation')
        df = df_bpy.merge(df_inflation, left_on = ['rate_dt'], right_index=True, how = 'inner')
        
    df = df.dropna(subset = ['10 YEARS'])

    return df    
