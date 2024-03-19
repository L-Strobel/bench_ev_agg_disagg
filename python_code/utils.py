"""
Utilities.
"""
import math
import python_code.config as c
import python_code.optimal as optimal
import python_code.profile_generator as ev
import python_code.cost_ts as cts
import python_code.opt_utils as outils

def to_ts(rslt_dict: dict[int, dict[int, float]], n_t: int) -> list[float]:
    """
    Calculate the total load.
    :param rslt_dict: Result in form dict: Agent ID -> load time series
    :param n_t: Number of time steps
    """
    ts = [0] * n_t
    for v in rslt_dict.values():
        for idx, val in v.items():
            ts[idx] += val
    return ts

def calc_total_costs(load: list[float], costs: list[float]) -> float:
    """
    Calculate the total costs of charging.
    :param load: Total load
    :param costs: Price time series
    """
    cost = 0
    for i, cc in enumerate(costs):
        cost += cc*load[i]
    return cost

def get_event_count(agents: list[ev.Agent]):
    """
    Get number of charging events with nonzero charged energy.
    :param agents: Agents
    """
    n_events = 0
    for agent in agents:
        events = outils.get_chrg_events_restricted(
            agent=agent, eta=c.ETA, soc_start=c.SOC_START, delta_t=c.DELTA_T
        )
        for event in events:
            if event.e_departure > event.e_arrival:
                n_events += 1
    return n_events

def prepare_run(
        trips, seed, n_agents, price_signal, fn_price,
        fn_gen, fn_dem
    ):
    """
    Load price signal, ev parameters, and mobility schedules.
    :param trips: MID trip data
    :param seed: RNG seed
    :param n_agents: Number of agents
    :param price_signal: Price time series
    :param fn_price: File name of price data
    :param fn_gen: File name of generation data
    :param fn_dem: File name of demand data
    """
    if price_signal == cts.PriceSignal.SINE:
        costs = cts.sinus_costs(c.N_DAYS)
    elif price_signal == cts.PriceSignal.REAL:
        costs = cts.real_costs(c.N_DAYS, seed, fn_price)
    elif price_signal == cts.PriceSignal.FUTURE:
        costs = cts.future_costs(c.N_DAYS, seed, fn_gen, fn_dem)

    # Process data
    agents = []
    for i in range(n_agents):
        agent = ev.Agent(
            id=i, events=[], capacity=c.EV_TYPE.value[0], car_type=c.EV_TYPE,
            p_home=c.P_HOME, p_work=c.P_WORK
        )
        agents.append(agent)
    agents = ev.get_charging_events(trips, agents, c.N_DAYS, seed=seed)

    return agents, costs

def get_opt(agents: list[ev.Agent], costs: list[float]):
    """
    Determine optimal costs.
    :param agents: Agents
    :param costs: Price time series
    """
    opt = calc_total_costs(
        to_ts(
            optimal.optimal(
                agents=agents, costs=costs, eta=c.ETA,
                soc_start=c.SOC_START, delta_t=c.DELTA_T, event_restricted=True),
            len(costs)
        ),
        costs
    )
    return opt


def print_progress(runs_done: int, n_runs: int):
    """
    Print progress.
    :param runs_done: Number of completed runs
    :param n_runs: Number of total runs
    """
    if runs_done % math.ceil(n_runs / 100) == 0:
        print(f"Progress: {runs_done/n_runs*100:.0f} %", flush=True)

