"""
Representative profile implementation.
"""
import random
import python_code.profile_generator as ev
import python_code.utils as utils
import python_code.optimal as optimal
from . import disaggregation as disagg

def pipeline(agents: list[ev.Agent], seed: int, n_profiles: int,
             disagg_option: disagg.Option, eta: float, soc_start: float,
             delta_t: float, costs: list[float]):
    """
    Run the pipeline for representative profiles
    :param agents: Agents
    :param seed: Rng seed
    :param n_profiles: Number of profiles
    :param disagg_option: Disaggregation method
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :param delta_t: Time step length [hour]
    :param costs: Cost time series
    """
    # Get representative profiles
    rng = random.Random(seed)
    n_profiles = min(len(agents), n_profiles)
    representatives = rng.sample(agents, k=n_profiles)
    scale_factor = len(agents) / n_profiles

    # Optimize
    load = utils.to_ts(
            optimal.optimal(
            agents=representatives, costs=costs, eta=eta,
            soc_start=soc_start, delta_t=delta_t, event_restricted=True
        ),
        len(costs)
    )
    for i, val in enumerate(load):
        load[i] = val * scale_factor

    # Disaggregation
    priority_metric = None
    if disagg_option == disagg.Option.LAXITY:
        priority_metric = disagg.prio_laxity
    elif disagg_option == disagg.Option.DEPARTURE:
        priority_metric = disagg.prio_departure
    return disagg.priority_based(
        agents=agents, power_signal=load, eta=eta, soc_start=soc_start,
        delta_t=delta_t, priority_metric=priority_metric
    )
