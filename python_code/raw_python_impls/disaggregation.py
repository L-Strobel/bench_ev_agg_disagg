"""
Disaggregate power signal to EVs. Used for virtual battery and representative profiles.
"""
from enum import Enum
from typing import Callable

import numpy as np

import python_code.opt_utils as utils
import python_code.profile_generator as ev

# Priority metrics
class Option(Enum):
    """
    Disaggregation option.
    """
    LAXITY = 0
    DEPARTURE = 1

def must_charge(event: utils.ChrgEvent, e: float, t: int, eta: float, delta_t: float) -> bool:
    """
    Determine if vehicle must charge immediately.
    :param event: Charging event
    :param e: State of energy currently
    :param t: Time step"
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    """
    t_flex = event.stop - t
    t_chrg = (event.e_departure - e)/(event.p_max*eta*delta_t)
    if (t_flex - t_chrg) < 1:
        return True
    return False

def prio_departure(event: utils.ChrgEvent, e: float, t: int, eta: float, delta_t: float) -> bool:
    """
    Earliest departure metric.
    :param event: Charging event
    :param e: State of energy currently
    :param t: Time step"
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    """
    if must_charge(event, e, t, eta, delta_t):
        return -np.inf

    return event.stop

def prio_laxity(event: utils.ChrgEvent, e: float, t: int, eta: float, delta_t: float) -> bool:
    """
    Least laxity metric.
    :param event: Charging event
    :param e: State of energy currently
    :param t: Time step"
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    """
    if must_charge(event, e, t, eta, delta_t):
        return -np.inf

    return event.stop - t - ((event.e_departure - e) / (event.p_max * eta * delta_t))

def priority_based(
        *, agents: list[ev.Agent], power_signal: list[float], eta: float,
        soc_start: float, delta_t: float, priority_metric: Callable
    ) -> dict[int, dict[int, float]]:
    """
    Disaggregate the power signal to the agents with a given priority metric.
    :param agents: Agents
    :param power_signal: Power signal
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :param delta_t: Time step length [hour]
    :param priority_metric: Priority metric
    """
    n_t = len(power_signal)
    demands = {agent.id: {} for agent in agents}

    # Get details
    active_events, e_bats = {}, {}
    consumptions = {agent.id: [0]*n_t for agent in agents}
    for agent in agents:
        e_bats[agent.id] = soc_start * agent.capacity # Starting SOC
        slack, _ = utils.get_slack(
            agent=agent, events=agent.events, e_bat_start=e_bats[agent.id], eta=eta, delta_t=delta_t
        )
        chrg_events = utils.get_chrg_events_restricted(
            agent=agent, eta=eta, soc_start=soc_start, delta_t=delta_t
        )

        # Determine what event is active at any given time
        active_events_agent = [None] * n_t
        for event in chrg_events:
            for t in range(event.start, event.stop):
                active_events_agent[t] = event
        active_events[agent.id] = active_events_agent

        # Consumption
        for i, event in enumerate(agent.events):
            consumptions[agent.id][event.stop-1] += event.consumption - slack[i]

    # Distribute power signal to agents
    remaining_power_signal = power_signal.copy()
    agent_by_id = {agent.id: agent for agent in agents}
    for t in range(n_t):
        # Get priority
        priority = {}
        for agent in agents:
            event = active_events[agent.id][t]
            if (event is not None) and (event.p_max != 0):
                priority[agent.id] = priority_metric(event, e_bats[agent.id], t, eta, delta_t)

        # Dispatch
        for agid, _ in sorted(priority.items(), key=lambda item: item[1]):
            agent = agent_by_id[agid]

            # Get time flexibility
            event = active_events[agent.id][t]
            tf = event.stop - t - ((event.e_departure-e_bats[agent.id]) / (event.p_max*eta*delta_t))

            # Determine maximum power agent can deliver
            ## Case 1: Signal fullfilled and no must charging
            pmin = 0
            if tf >= 1 and remaining_power_signal[t] <= 0:
                continue
            ## Case 2: Necessary charging
            if tf <= 0:
                pmin = event.p_max
            elif tf < 1:
                pmin = (1 - tf) * event.p_max
            ## Case 3: Flexible charging to fullfill power signal
            pmax = max(pmin, min(remaining_power_signal[t], event.p_max))

            # Charge
            e_charge = max(0, min(pmax * eta * delta_t, agent.capacity - e_bats[agent.id]))
            p_charge = e_charge / (eta * delta_t)
            e_bats[agent.id] += e_charge

            # Update power signal
            remaining_power_signal[t] -= p_charge

            # Save
            if p_charge > 0:
                demands[agent.id][t] = p_charge

        # Consumption
        for agent in agents:
            e_bats[agent.id] -= consumptions[agent.id][t]
    return demands
