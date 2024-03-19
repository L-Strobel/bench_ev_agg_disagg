"""
Virtual battery implementation.
"""
import gurobipy as grb

import python_code.utils as utils
import python_code.opt_utils as opt_utils
import python_code.profile_generator as ev
from . import disaggregation as disagg
from . import fo_grouping as fogrp


# pylint fails to look into gurobipy
# pylint: disable=no-member

def get_vb_params(
        chrg_events: list[opt_utils.ChrgEvent], n_t: int, eta: float, delta_t: float
    ):
    """
    Determine parameters for vb optimization.
    :param chrg_events: Charging events
    :param n_t: Number of time steps
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    """
    capability  = [0] * n_t
    e_arrival   = [0] * n_t
    p_avg       = [0] * n_t
    p_min       = [0] * n_t
    e_bat_min   = [0] * (n_t+1)
    e_bat_max   = [0] * (n_t+1)
    e_departure = [0] * (n_t+1)

    for event in chrg_events:
        e_arrival[event.start] += event.e_arrival
        e_departure[event.stop] += event.e_departure

        for t in range(event.start, event.stop):
            e_needed = event.e_departure - event.e_arrival
            p_avg[t] += e_needed / ((event.stop - event.start) * delta_t * eta)
            capability[t] += event.p_max

            # Event without flexibility
            if e_needed >= ((event.stop - event.start) * delta_t * eta * event.p_max):
                p_min[t] += event.p_max
            else:
                p_min[t] += 0

        for t, e in event.calc_e_bat_min(eta, delta_t).items():
            e_bat_min[t] += e

        for t, e in event.calc_e_bat_max(eta, delta_t).items():
            e_bat_max[t] += e
    return capability, e_bat_min, e_bat_max, e_arrival, e_departure, p_avg, p_min

def optimize(
        *, agents: list[ev.Agent], costs: list[float], eta: float,
        soc_start: float, delta_t: float, support_point: tuple[float, float],
        offset_at_full: float, with_fsoe: bool
    ) -> list[float]:
    """
    Optimize agents as a virtual battery.
    :param agents: Agents
    :param costs: Price time series
    :param eta: Charging efficiency
    :param soc_start: Starting SOC
    :param delta_t: Time step length [hour]
    :param support_point: Middle point that defines the piecewise-linear function P_max = f(soe)
    Points are: [(0, 1), support_point, (capability, offset_at_full)
    :param offset_at_full: Allowed charging power if virtual battery is almost fully charged.
    :param with_fsoe: With P_max = f(soe) constraint?
    """
    # Get charging events from data
    chrg_events = []
    for agent in agents:
        agent_chrg_events = opt_utils.get_chrg_events_restricted(
            agent=agent, eta=eta, soc_start=soc_start, delta_t=delta_t
        )
        chrg_events.extend( agent_chrg_events )

    demand = optimize_from_events(
        chrg_events=chrg_events, costs=costs, eta=eta, delta_t=delta_t,
        support_point=support_point, offset_at_full=offset_at_full, with_fsoe=with_fsoe
    )
    return demand

def optimize_from_events(
        *, chrg_events: list[opt_utils.ChrgEvent], costs: list[float],
        eta: float, delta_t: float, support_point: tuple[float, float],
        offset_at_full: float, with_fsoe: bool
    ) -> list[float]:
    """
    Optimize agents as a virtual battery.
    :param chrg_events: Charging events
    :param costs: Price time series
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :param support_point: Middle point that defines the piecewise-linear function P_max = f(soe)
    Points are: [(0, 1), support_point, (capability, offset_at_full)
    :param offset_at_full: Allowed charging power if virtual battery is almost fully charged.
    :param with_fsoe: With P_max = f(soe) constraint?
    """
    n_t = len(costs)

    # Get parameters from chargin events
    params = get_vb_params(chrg_events, n_t, eta, delta_t)
    capability, e_bat_min, e_bat_max, e_arrival, e_departure, p_avg, p_min = params

    # Optimization
    demand = [0] * n_t
    with grb.Env(params={"OutputFlag": 0}) as env, grb.Model(env=env) as model:
        # Variables
        p_charge =  {} # Charging power of EV
        for t in range(n_t):
            # p_min[t] <= p_charge <= capability[t]
            p_charge[t] = model.addVar(lb=p_min[t], ub=capability[t])

        e_bat = {} # Battery energy content
        e_bat[0] = 0 # Initially empty. Energy content of EVs at t=0 is included in arrivalEnergy
        for t in range(1, n_t+1):
            # e_bat_min[t] <= e_bat <= e_bat_max[t]
            e_bat[t] = model.addVar(lb=e_bat_min[t], ub=e_bat_max[t])

        # Constraints
        for t in range(n_t):
            # Energy continuity
            # E-Battery(t+1) = Charged-Energy + Energy-Arriving - Energy-Departing + E-Battery(t)
            e_charged = p_charge[t] * eta * delta_t + e_arrival[t]
            model.addLConstr(lhs= e_charged - e_departure[t] + e_bat[t],
                             sense=grb.GRB.EQUAL,
                             rhs=e_bat[t+1])

        # Add PWL constraint p_max = f(soe)
        if with_fsoe:
            p_max = {}
            for t in range(1, n_t):
                # Ensure feasibility
                y_mid = max(p_avg[t], capability[t]*support_point[1])
                y_right = max(p_avg[t], capability[t]*offset_at_full)
                # Add constraint
                p_max[t] = model.addVar()
                model.addLConstr(lhs=p_charge[t],
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=p_max[t])
                model.addGenConstrPWL(
                    e_bat[t], p_max[t],
                    [0, e_bat_max[t]*support_point[0], e_bat_max[t]],
                    [capability[t], y_mid, y_right]
                )

        # Objective
        objective = grb.LinExpr()
        for t in range(n_t):
            objective.addTerms(costs[t], p_charge[t])
        model.ModelSense = grb.GRB.MINIMIZE
        model.setObjective(objective)

        # Calculation
        model.optimize()

        # Check Success
        if model.status == 13:
            print("Solution suboptimal!")
        elif model.status > 2:
            raise AssertionError(f"Model terminated! Status {model.status}")

        # Solution
        for t in range(n_t):
            demand[t] = p_charge[t].X
    return demand

def pipeline(
        *, agents: list[ev.Agent], costs: list[float], eta: float, soc_start: float,
        delta_t: float, disagg_option: disagg.Option,
        support_point: tuple[float, float], offset_at_full: float, with_fsoe: bool
    ) -> list[float]:
    """
    Run the entire pipeline for VB:
    Aggregation -> Optimization -> Disaggregation
    :param agents: Agents
    :param costs: Price time series
    :param eta: Charging efficiency
    :param soc_start: Starting SOC
    :param delta_t: Time step length [hour]
    :param disagg_option: Disaggregation option
    :param support_point: Middle point that defines the piecewise-linear function P_max = f(soe)
    Points are: [(0, 1), support_point, (capability, offset_at_full)
    :param offset_at_full: Allowed charging power if virtual battery is almost fully charged.
    :param with_fsoe: With P_max = f(soe) constraint?
    """
    power_signal = optimize(
        agents=agents, costs=costs, eta=eta, soc_start=soc_start, delta_t=delta_t,
        support_point=support_point, offset_at_full=offset_at_full, with_fsoe=with_fsoe
    )

    # Disaggregation
    priority_metric = None
    if disagg_option == disagg.Option.LAXITY:
        priority_metric = disagg.prio_laxity
    elif disagg_option == disagg.Option.DEPARTURE:
        priority_metric = disagg.prio_departure

    return disagg.priority_based(
        agents=agents, power_signal=power_signal, eta=eta, soc_start=soc_start,
        delta_t=delta_t, priority_metric=priority_metric
    )

def pipeline_grpd(
        *, agents: list[ev.Agent], costs: list[float], eta: float, soc_start: float,
        delta_t: float, est: float, lst: float, disagg_option: disagg.Option,
        support_point: tuple[float, float], offset_at_full: float, with_fsoe: bool
    ) -> tuple[list[float], int]:
    """
    Run the entire pipeline for VB with grouping:
    Grouping -> Aggregation -> Optimization -> Disaggregation
    :param agents: Agents
    :param costs: Price time series
    :param eta: Charging efficiency
    :param soc_start: Starting SOC
    :param delta_t: Time step length [hour]
    :param est: earliest start time tolerance
    :param lst: latest stop time tolerance
    :param disagg_option: Disaggregation option
    :param support_point: Middle point that defines the piecewise-linear function P_max = f(soe)
    Points are: [(0, 1), support_point, (capability, offset_at_full)
    :param offset_at_full: Allowed charging power if virtual battery is almost fully charged.
    :param with_fsoe: With P_max = f(soe) constraint?
    """
    n_t = len(costs)
    demand = [0] * n_t

    # Charging events
    chrg_events = []
    for agent in agents:
        agent_chrg_events = opt_utils.get_chrg_events_restricted(
            agent=agent, eta=eta, soc_start=soc_start, delta_t=delta_t
        )
        chrg_events.extend(agent_chrg_events)

    # Group
    groups = fogrp.pre_group(chrg_events, est, lst, lambda x: x.start, lambda x: x.stop)
    groups = fogrp.optimize_groups(groups, est, lst, lambda x: x.start, lambda x: x.stop)

    # Optimize
    power_signal = [0] * n_t
    for group in groups:
        grp_events = [o for c in group.cells for o in c.objects]
        p = optimize_from_events(
            chrg_events=grp_events, costs=costs, eta=eta, delta_t=delta_t,
            support_point=support_point, offset_at_full=offset_at_full, with_fsoe=with_fsoe
        )
        for t, val in enumerate(p):
            power_signal[t] += val

    # Disaggregate
    priority_metric = None
    if disagg_option == disagg.Option.LAXITY:
        priority_metric = disagg.prio_laxity
    elif disagg_option == disagg.Option.DEPARTURE:
        priority_metric = disagg.prio_departure
    y = utils.to_ts(
        disagg.priority_based(
            agents=agents, power_signal=power_signal, eta=eta, soc_start=soc_start,
            delta_t=delta_t, priority_metric=priority_metric),
        n_t
    )

    # Add to result
    for i, v in enumerate(y):
        demand[i] += v
    return demand, len(groups)
