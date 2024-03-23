"""
Cost time series
"""
from enum import Enum
import random
import numpy as np
import pandas as pd

class PriceSignal(Enum):
    """
    Implemented price signals
    """
    SINE = 0
    REAL = 1
    FUTURE = 2

def sinus_costs(ndays: int) -> np.ndarray:
    """
    Sinus cost time series
    :param ndays: Numner of days 
    """
    n = ndays * 96
    return np.cos(2*np.pi/96*np.arange(n))

def real_costs(ndays: int, seed: int, fn: str) -> np.ndarray:
    """
    Real spot market cost time series
    :param ndays: Numner of days 
    :param seed: Random seed
    :param fn: Spot market price data from www.smard.de
    """
    rng = random.Random(seed)

    # Load data
    spot_prices = pd.read_csv(
        fn, decimal=".", sep=";",
        na_values="-", thousands=",", parse_dates=["Date"]
    )
    spot_prices["weekday"] = spot_prices.Date.dt.isocalendar()["day"]

    # Choose random start point
    starte_dates = list(spot_prices[
        (spot_prices.weekday == 1) & (spot_prices.index + 96*ndays <= 96*365)
    ].Date.unique())
    start_date = rng.choice(starte_dates)
    start_idx = spot_prices[
        (spot_prices.Date == start_date) & (spot_prices.Start == "12:00 AM")
    ].index[0]

    # Normalize
    costs = spot_prices.loc[
        start_idx: start_idx+ndays*96-1, 'Germany/Luxembourg [€/MWh] Calculated resolutions'
    ].values
    costs = costs / costs.max()
    return costs

def future_costs(ndays: int, seed: int, fn_gen: str, fn_dem: str) -> np.ndarray:
    """
    Stand in for future spot market cost time series
    :param ndays: Numner of days
    :param seed: Random seed
    :param fn_gen: Electricity generation data from www.smard.de
    :param fn_dem: Electricity demand data from www.smard.de
    """
    rng = random.Random(seed)

    # Load data
    generation = pd.read_csv(
        fn_gen, decimal=".", sep=";",
        na_values="-", thousands=",", parse_dates=["Date"]
    )
    load = pd.read_csv(
        fn_dem, decimal=".", sep=";",
        na_values="-", thousands=",", parse_dates=["Date"]
    )
    load["weekday"] = load.Date.dt.isocalendar()["day"]

    # Calculate residual load. Based on NEP2023 scenario B
    load["Residual_load_synthetic_2045"] = (
        load["Total (grid load) [MWh] Original resolutions"] * (1025 / 478)
        - generation["Wind offshore [MWh] Original resolutions"] *  (70 / 7.8)
        - generation["Wind onshore [MWh] Original resolutions"] * (160 / 56.1)
        - generation["Photovoltaics [MWh] Original resolutions"] * (400 / 59.3)
    )

    # Choose random start point
    starte_dates = list(load[
        (load.weekday == 1) & (load.index + 96*ndays <= 96*365)
    ].Date.unique())
    start_date = rng.choice(starte_dates)
    start_idx = load[(load.Date == start_date) & (load.Start == "12:00 AM")].index[0]

    # Normalize
    costs = load.loc[start_idx: start_idx+ndays*96-1, 'Residual_load_synthetic_2045'].values
    costs = costs / costs.max()
    return costs
