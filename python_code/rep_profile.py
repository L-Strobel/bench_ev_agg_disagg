"""
Representative profile implementation.
"""
import random

# pylint: disable=import-error
import ev_agg_bench_rs as rs # type: ignore
# pylint: enable=import-error

import python_code.profile_generator as ev
import python_code.optimal as optimal
import python_code.utils as utils

def pipeline(agents: list[ev.Agent], seed: int, n_profiles: int,
             disagg_option: str, eta: float, soc_start: float,
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
    events = ev.get_chrg_events_restricted(agents, eta, soc_start, delta_t)
    load = rs.disaggregate(events, load, eta, delta_t, disagg_option)
    return load
