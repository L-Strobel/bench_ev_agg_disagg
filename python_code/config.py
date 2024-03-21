"""
Global config
"""
import python_code.profile_generator as ev

N_DAYS = 7
SOC_START = 0.9
DELTA_T = 1/4
ETA = 0.9
EV_TYPE = ev.CarType.BEV_MEDIUM
P_HOME = 11
P_WORK = 0
EPS_GEOM = 1e-5

# Paths: CHANGE ME
MID_LOCATION = "/path/to/wege.csv" # Path to wege dataset of MID
# Downloadable from smard.de
FN_PRICE = "../input_data/Day-ahead_prices_202201010000_202301012359_Quarterhour.csv"
FN_GEN = "../input_data/Actual_generation_202201010000_202301012359_Quarterhour.csv"
FN_DEM = "../input_data/Actual_consumption_202201010000_202301012359_Quarterhour.csv"
