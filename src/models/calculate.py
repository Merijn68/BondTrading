
import pandas as pd


def estimate_bond_price (       
    frequency: int,
    ytm: float,
    coupon_rate: float,
    years_to_maturity: int,
    par_value: float = 100,   
) -> float:
    '''
        Estimation of present value of future cashflows...  
        Loosly based on https://www.wallstreetmojo.com/bond-pricing-formula/
              
    '''
    if coupon_rate == 0:
        # Zero Coupon        
        p = par_value / ( 1 + ytm) ^ years_to_maturity
    else:
        no_of_periods = years_to_maturity * frequency
        coupon = (coupon_rate / frequency * par_value)         
        r = coupon  * (1 - (1 + ytm/coupon_rate) ^ (- no_of_periods )) / ( ytm / frequency)
        c = par_value / ((1+ ytm/frequency) ^ (no_of_periods)) 
        p = r + c
    return p
