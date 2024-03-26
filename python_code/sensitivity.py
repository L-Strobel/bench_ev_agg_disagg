"""
Run multiple different configs
"""
import pandas as pd
from io import StringIO
from copy import deepcopy

import python_code.benchmark as benchmark
import python_code.config

if __name__ == "__main__":
    REPS = 10

    # Load MID data, change this path to your copy of "Wege.csv" of the MID 2017
    import midcrypt
    mid_trips = midcrypt.fetchFromDir("/home/leo/J/MID/MiD2017_Lokal_Wege.csv.encrypted")
    mid_trips = pd.read_csv(StringIO(mid_trips), sep=";", decimal=",")

    # Define runs
    configs = []
    defaultConfig = python_code.config.defaultConfig

    # Work charging
    work_charging = deepcopy(defaultConfig)
    work_charging.P_WORK = 11
    work_charging.name = "WORK"
    configs.append(work_charging)

    # Winter consumption
    winter = deepcopy(defaultConfig)
    winter.EV_TYPE.value[1] = 0.22
    winter.name = "WINTER"
    configs.append(winter)

    # Power high
    pHigh = deepcopy(defaultConfig)
    pHigh.P_HOME = 22
    pHigh.name = "P_HIGH"
    configs.append(pHigh)

    # Power low
    pLow = deepcopy(defaultConfig)
    pLow.P_HOME = 4.7
    pLow.name = "P_LOW"
    configs.append(pLow)

    # Run
    for config in configs:
        benchmark.run(
            mid_trips, REPS, python_code.config.FN_PRICE, python_code.config.FN_GEN,
            python_code.config.FN_DEM, config
            )
        