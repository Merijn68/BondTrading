
from re import M
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
        
        F = par_value
        C = coupon_rate * F / frequency
        n = years_to_maturity * frequency
        r = ytm
        pv_F = F / ( 1 + r) ** n
        pc_C = C / ( 1 + r) ** n

        


        C = 1 - 1+ytm
         = pow(((1 + ytm) / coupon_rate) , (- no_of_periods ))
        r = coupon  * (1 - C) / ( ytm / frequency)
        print ('r', r)
        c = par_value / ((1+ ytm/frequency) ** (no_of_periods)) / 100
        print ('c', c)
        p = r + c
    return p
