"""
    Pre process data from raw to processed data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from loguru import logger
from src.features import build_features
from src.data import join_data


def read_csv(
    path: Path = Path("../data/raw/bonds.csv"), thousands: str = ",", *args, **kwargs
) -> pd.DataFrame:
    """read a csv file with some logging"""

    logger.info(f"Loading data from {path}")
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path, thousands=thousands, dayfirst=True, *args, **kwargs)
    except Exception as error:
        logger.error(f"Error loading data: {error}")
    return df


def get_bond_data(
    path: Path = Path("../data/raw/bonds.csv"),
) -> pd.DataFrame:
    logger.info("Load bond data")
    """ Read raw bond data from CSV, drop columns and format data """

    # loading bond data
    df = read_csv(path, parse_dates=["issue_dt", "first_coupon_date", "mature_dt"])
    # remove special character
    df.columns = df.columns.str.replace(" ", "")

    # Drop unneeded columns
    df = df.drop(
        ["bondname", "group_name", "fix_float", "cparty_type", "CO2_factor"], axis=1
    )

    # Correct columns types
    for column in [
        "cfi_code",
        "isin",
        "ccy",
        "issuer_name",
        "coupon_frq",
        "country_name",
        "issue_rating",
        "bond_ext_name",
    ]:
        df[column] = df[column].astype("string")

    # Remove trailing spaces
    for col in ["isin", "coupon_frq", "country_name", "issuer_name"]:
        df[col] = df[col].str.strip()

    # rename country_name to country
    df = df.rename(columns={"country_name": "country"})

    # Total Issue amount '-' should be converted to 0
    df.loc[df["tot_issue"] == "-", "tot_issue"] = "0"
    df["tot_issue"] = (
        df["tot_issue"].str.replace(",", "").str.replace("-", "0").astype("float")
        / 1000000
    )

    df["coupon"] = (
        df["coupon"].str.replace(",", "").str.replace("-", "0").astype("float")
    )

    return df


def impute_bonds(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Impute bond data")

    # CFI Codes Imputed with Dept Instrument XXXXX (Unknown)
    df["cfi_code"] = df["cfi_code"].fillna("DXXXXX")

    # Date 1899-12-30 means no first coupon date was known...
    df.loc[df["first_coupon_date"] == "30-12-1899", "first_coupon_date"] = None
    df.loc[df["issue_dt"] == "30-12-1899", "issue_dt"] = None
    df.loc[df["mature_dt"] == "30-12-1899", "mature_dt"] = None

    # impute first coupon date if missing
    df["first_coupon_date"] = df["first_coupon_date"].fillna(df["issue_dt"])

    # Bond duration (estimation in dagen)
    df["bond_duration"] = df["mature_dt"] - df["issue_dt"]
    df["bond_duration"] = df["bond_duration"].dt.days

    # Fill missing issuer_rating met meest voorkomende waarde per issuer name
    # voor nu verwijderen we deze bonds
    df = df[~df["issue_rating"].isnull()].copy()

    # Nice name for presentations
    df["issue"] = (
        df["isin"]
        + " "
        + df["country"]
        + " "
        + df["bond_ext_name"].str.split(n=1).str[1]
    )

    return df


def get_price(
    path: Path = Path("../data/raw/price.csv"),
) -> pd.DataFrame:
    """load raw bond price data"""

    logger.info("Load bond price data")

    df = read_csv(path, parse_dates=["rate_dt"])

    df.columns = df.columns.str.replace(" ", "")  # remove spaces in column names

    # Correct columns types
    for column in ["reference_identifier", "ccy"]:
        df[column] = df[column].astype("string")

    # Bid and Offer are not needed as they are the same in data
    # Merge this in a mid price
    df["mid"] = (df["bid"] + df["offer"]) / 2
    df = df.drop(["bid", "offer"], axis=1)

    # Data for 30-7-2021 is duplicated - remove this
    df = df.drop_duplicates()
    return df


def impute_price(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Impute raw bond price data"""

    logger.info("Impute bond price")

    # Last day of the year contains a duplicate of the previous day - drop this
    s = pd.date_range("2010-01-01", periods=12, freq="BY")
    df = df[~df["rate_dt"].isin(s)].copy()

    # 99.999 is a 'default' waarde. Remove this
    df = df[df["mid"] != 99.999]

    return df


def get_yield(
    path: Path = Path("../data/raw/yield.csv"),
) -> pd.DataFrame:
    """load raw yield data"""

    logger.info("Load goverment yield curve data")

    country_dict = {
        "DE": "Germany",
        "FR": "France",
        "ES": "Spain",
        "IT": "Italy",
        "NL": "Netherlands",
    }

    df = read_csv(path, parse_dates=["rate_dt", "actual_dt"])

    df.columns = df.columns.str.replace(" ", "")  # remove spaces from column names

    # Add country column
    for country in country_dict:
        df.loc[df["ratename"].str.contains(country), "country"] = country_dict[country]

    # Data contains multiple rows per day in exceptional cases. Keep only last
    df = df.groupby(["country", "rate_dt", "timeband"]).nth(-1)
    df = df.reset_index()

    for column in ["ratename", "ccy", "timeband", "int_basis", "country"]:
        df[column] = df[column].astype("string")
        df[column] = df[column].str.strip()

    for column in ["bid", "offer"]:
        df[column] = df[column].replace(",", "").replace("-", "0").astype("float")

    return df


def impute_yield(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Impute yield data"""

    logger.info("Impute yield curve")

    # Drop MM curves only Bond Based curves are correct and usefull
    df = df[df["ratename"].str.contains("BB")].copy()

    # Recalculate actual dt -> to keep in line with inflatie data
    # We do not take holidays into account
    df["offset"] = df["timeband"].str.extract(r"(\d+)")
    df["actual_dt"] = df["rate_dt"] + df["offset"].astype("timedelta64[Y]")
    df = df.drop(columns="offset")

    df["time"] = (
        df["actual_dt"] - df["rate_dt"]
    ).dt.days  # Timeband transformed into number of days
    df["mid"] = (df["bid"] + df["offer"]) / 2  # Calculate mid rate
    df[df["rate_dt"] >= "1-jan-2010"]  # drop any data before 1-jan-2010

    return df


def get_inflation(
    countrydict: dict = {
        "DE": "Germany",
        "FR": "France",
        "ES": "Spain",
        "IT": "Italy",
        "US": "United States",
    },
    path: Path = Path("../data/raw"),
) -> pd.DataFrame:
    """load inflation data"""

    logger.info("Load goverment yield curve data")

    df = pd.DataFrame()
    for country in countrydict:
        filepath = Path(path, f"{country} Inflation.csv")
        dfc = read_csv(filepath, skiprows=2, header=None)
        if not dfc.empty:
            dfc = dfc.iloc[:, 2:17]
            dfc.columns = [
                "rate_dt",
                "1 YEAR",
                "2 YEARS",
                "3 YEARS",
                "4 YEARS",
                "5 YEARS",
                "6 YEARS",
                "7 YEARS",
                "8 YEARS",
                "9 YEARS",
                "10 YEARS",
                "15 YEARS",
                "20 YEARS",
                "25 YEARS",
                "30 YEARS",
            ]
            dfc["country"] = countrydict[country]

            df = pd.concat([df, dfc])

    df = df.melt(
        id_vars=["country", "rate_dt"], var_name="timeband", value_name="inflation"
    )
    df["ratename"] = "Inflation"

    for column in ["ratename", "timeband", "country"]:
        df[column] = df[column].astype("string")
    for column in ["rate_dt"]:
        df[column] = pd.to_datetime(df[column])
    return df


def impute_inflation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Impute raw inflation data"""
    logger.info("Impute inflation curve")

    df = df[~df["inflation"].isnull()].copy()  # Drop all NANs
    # Translate timeband to actual_dt
    df["offset"] = df["timeband"].str.extract(r"(\d+)")
    df["actual_dt"] = df["rate_dt"] + df["offset"].astype("timedelta64[Y]")
    df = df.drop(columns="offset")
    # De timeband omzetten naar aantal dagen.
    df["time"] = (df["actual_dt"] - df["rate_dt"]).dt.days

    return df


def make_isin(
    df_bp: pd.DataFrame,
    df_yield: pd.DataFrame,
    df_inflation: pd.DataFrame,
    isin: str = "NL0011220108",  # 10 Years NL Bond, maturity 2025 0.25% coupon
) -> pd.DataFrame:
    """Preprocess bonddata for a specific bond for analyses on mulitple features"""
    logger.info(f"Create dataset for bond {isin}")

    df_isin = df_bp[df_bp["reference_identifier"] == isin]
    df_isin = join_data.join_yield(df_isin, df_yield)
    df_isin = join_data.yield_to_maturity(df_isin, df_yield)
    df_isin = build_features.add_estimated_bond_price(df_isin)
    df_isin = build_features.add_term_spread(df_isin)
    df_isin = build_features.add_bid_offer_spread(df_isin)
    df_isin = join_data.join_10y_inflation(df_isin, df_inflation, country="Germany")
    return df_isin


def make_data():
    """Generate all data preprocessing"""

    # Bond header and characteristics
    df_bonds = get_bond_data()
    df_bonds = impute_bonds(df_bonds)
    save_pkl("bonds", df_bonds)

    # Pricing data on daily basis
    df_price = get_price()
    df_price = impute_price(df_price)
    save_pkl("price", df_price)

    # Government yield rates on daily basis
    df_yield = get_yield()
    df_yield = impute_yield(df_yield)
    save_pkl("yield", df_yield)

    # Inflation curves on daily basis
    df_inflation = get_inflation()
    df_inflation = impute_inflation(df_inflation)
    save_pkl("inflation", df_inflation)

    # Join Bond price with bond characteristics
    df_bp = join_data.join_price(df_bonds, df_price)
    df_bp = build_features.add_duration(df_bp)
    save_pkl("bp", df_bp)

    # ISIN preprocessed data for a single bond
    df_isin = make_isin(df_bp, df_yield, df_inflation)

    save_pkl("isin", df_isin)


def save_pkl(name: str, df: pd.DataFrame, protocol: int = 4):
    """Store processed data"""
    logger.info(f"Save preprocessed {name} data")
    try:

        df.to_pickle(f"../data/processed/{name}.pkl", protocol=protocol)

        # Files are stored both in pickle and csv format
        # Attribute metadata is added in json format to the csv for ease of processing
        data = {}
        dtype = {}
        for column in df.columns:
            if column in df.select_dtypes(include=[np.datetime64, np.float]).columns:
                dtype[column] = "float"
            elif column in df.select_dtypes(include=[np.int64, np.integer]).columns:
                dtype[column] = "int"
            else:
                dtype[column] = "string"
        data["dtype"] = dtype
        data["parse_dates"] = df.select_dtypes(include=[np.datetime64]).columns.tolist()

        f = open(f"../data/processed/{name}.json", "w")
        json.dump(data, f)
        f.close()
        df.to_csv(f"../data/processed/{name}.csv", index=False)

    except Exception as error:
        logger.error(f"Error saving {name} data: {error}")


def read_pkl(
    name: str, path=Path("../data/processed/"), colab: bool = False
) -> pd.DataFrame:
    """load processed data"""
    logger.info(f"Load preprocessed {name} data")

    df = pd.DataFrame()
    if not path.is_dir():
        logger.error(f"Directory {path} not found")
    else:

        if colab:
            # Google Colab was not able to process pickled data
            # Alternatively I store and load attribute information in JSon format
            filepath = Path(path, name + ".json")
            if filepath.exists():
                f = open(Path(path, f"{name}.json"))
                data = json.load(f)
                f.close
            filepath = Path(path, name + ".csv")
            if filepath.exists():
                if data:
                    df = pd.read_csv(
                        filepath, parse_dates=data["parse_dates"], dtype=data["dtype"]
                    )
                else:
                    logger.info(f"Metadata missing for file {name}")
                    df = pd.read_csv(filepath, data)
            else:
                logger.error(f"File {path} not found")
        else:
            path = Path(path, name + ".pkl")
            if not path.exists():
                logger.error(f"File {path} not found")
            else:
                df = pd.read_pickle(path)

    return df
