import pandas as pd
from loguru import logger


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add the remainind duration (from rate_dt to maturity_dt)"""

    logger.info("Add remaining duration...")
    df["remain_duration"] = df["mature_dt"] - df["rate_dt"]
    df["remain_duration"] = df["remain_duration"].dt.days
    return df


def add_term_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate difference between 10 years yield and 2 years yield"""

    logger.info("Add term spread...")
    mid_10y = df["y_bid10"] + df["y_offer10"] / 2
    mid_2y = df["y_bid2"] + df["y_offer2"] / 2
    df["term_spread"] = mid_10y - mid_2y
    return df


def add_bid_offer_spread(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Add bid offer spread...")
    """ For each tenor calculate difference between bid and ask yield """

    years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    bids = ["".join(("y_bid", str(year))) for year in years]
    offers = ["".join(("y_offer", str(year))) for year in years]
    df_spread = pd.DataFrame(df[bids].to_numpy() - -df[offers].to_numpy())
    columns = ["".join(("bid_offer_spread_", str(year))) for year in years]
    df_spread.columns = columns
    df = pd.concat([df, df_spread], axis=1)

    return df


def add_estimated_bond_price(
    df: pd.DataFrame, var: str = "estimated_bondprice"
) -> pd.DataFrame:
    """Calculate theoreritcal bond price based on yield to maturity"""

    df[var] = df.apply(calculate_price, axis=1)
    return df


def calculate_price(row) -> float:
    """Calculate Theoretical Bond price"""

    periods = row["remain_duration"] / 365  # Actual / 365 approx
    ytm = row["ytm"] / 100
    coupon = row["coupon"] / 100
    if row["coupon_frq"] == "ANNUAL":
        frequency = 1
    else:
        frequency = 2
    par_value = 100
    i = coupon / frequency * par_value  # Interest amount
    p = (1 - (1 + ytm / frequency) ** (-periods)) / (
        ytm / frequency
    )  # compounded interest
    pi = p * i  # Interest on coupons
    k = par_value / ((1 + ytm / frequency) ** periods)  # Interest on capital
    price = k + pi
    return price
