# pylint: disable=C
"""
Global config
"""
from dataclasses import dataclass
import python_code.profile_generator as ev

@dataclass
class Config:
    """
    Config for benchmark. Tested changes: EV_TYPE, P_HOME, P_WORK
    """
    name: str
    N_DAYS: int
    SOC_START: float
    DELTA_T: float
    ETA: float
    EV_TYPE: ev.CarType
    P_HOME: float
    P_WORK: float
    EPS_GEOM: float

defaultConfig = Config(
    name = "DEFAULT",
    N_DAYS = 7, SOC_START = 0.9, DELTA_T = 1/4,
    ETA = 0.9, EV_TYPE = ev.CarType.BEV_MEDIUM,
    P_HOME = 11, P_WORK = 11, EPS_GEOM = 1e-5
)

# Paths: CHANGE ME
MID_LOCATION = "/path/to/wege.csv" # Path to wege dataset of MID
# Downloadable from smard.de
FN_PRICE = "../input_data/Day-ahead_prices_202201010000_202301012359_Quarterhour.csv"
FN_GEN = "../input_data/Actual_generation_202201010000_202301012359_Quarterhour.csv"
FN_DEM = "../input_data/Actual_consumption_202201010000_202301012359_Quarterhour.csv"
