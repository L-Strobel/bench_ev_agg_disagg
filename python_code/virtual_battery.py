"""
Virtual battery implementation.
"""
import gurobipy as grb

# pylint: disable=import-error
import ev_agg_bench_rs as rs # type: ignore
# pylint: enable=import-error

import python_code.opt_utils as opt_utils

# pylint fails to look into gurobipy
# pylint: disable=no-member

def optimize(
        *, events: list[opt_utils.ChrgEvent], costs: list[float],
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
    vbslices = rs.vb.get_vb_params(events, n_t, eta, delta_t)

    # Optimization
    demand = [0] * n_t
    with grb.Env(params={"OutputFlag": 0}) as env, grb.Model(env=env) as model:
        # Variables
        p_charge =  {} # Charging power of EV
        for t in range(n_t):
            # p_min[t] <= p_charge <= capability[t]
            p_charge[t] = model.addVar(lb=vbslices[t].p_min, ub=vbslices[t].capability)

        e_bat = {} # Battery energy content
        e_bat[0] = 0 # Initially empty. Energy content of EVs at t=0 is included in arrivalEnergy
        for t in range(1, n_t+1):
            # e_bat_min[t] <= e_bat <= e_bat_max[t]
            e_bat[t] = model.addVar(lb=vbslices[t].e_min, ub=vbslices[t].e_max)

        # Constraints
        for t in range(n_t):
            # Energy continuity
            # E-Battery(t+1) = Charged-Energy + Energy-Arriving - Energy-Departing + E-Battery(t)
            e_charged = p_charge[t] * eta * delta_t + vbslices[t].e_arrival
            model.addLConstr(lhs= e_charged - vbslices[t].e_departure + e_bat[t],
                             sense=grb.GRB.EQUAL,
                             rhs=e_bat[t+1])

        # Add PWL constraint p_max = f(soe)
        if with_fsoe:
            p_max = {}
            for t in range(1, n_t):
                # Ensure feasibility
                y_mid = max(vbslices[t].p_avg, vbslices[t].capability*support_point[1])
                y_right = max(vbslices[t].p_avg, vbslices[t].capability*offset_at_full)
                # Add constraint
                p_max[t] = model.addVar()
                model.addLConstr(lhs=p_charge[t],
                                sense=grb.GRB.LESS_EQUAL,
                                rhs=p_max[t])
                model.addGenConstrPWL(
                    e_bat[t], p_max[t],
                    [0, vbslices[t].e_max*support_point[0], vbslices[t].e_max],
                    [vbslices[t].capability, y_mid, y_right]
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
        *, events: list[rs.ChrgEvent], costs: list[float], eta: float, delta_t: float,
        disagg_option: str, support_point: tuple[float, float], offset_at_full: float,
        with_fsoe: bool
    ) -> list[float]:
    """
    Run the entire pipeline for VB:
    Aggregation -> Optimization -> Disaggregation
    :param events: Charging events
    :param costs: Price time series
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :param disagg_option: Disaggregation option
    :param support_point: Middle point that defines the piecewise-linear function P_max = f(soe)
    Points are: [(0, 1), support_point, (capability, offset_at_full)
    :param offset_at_full: Allowed charging power if virtual battery is almost fully charged.
    :param with_fsoe: With P_max = f(soe) constraint?
    """
    power_signal = optimize(
        events=events, costs=costs, eta=eta, delta_t=delta_t,
        support_point=support_point, offset_at_full=offset_at_full, with_fsoe=with_fsoe
    )

    # Disaggregation
    load = rs.disaggregate(events, power_signal, eta, delta_t, disagg_option)
    return load

def pipeline_grpd (
        *, events: list[rs.ChrgEvent], costs: list[float], eta: float, delta_t: float,
        est: float, lst: float, disagg_option: str,
        support_point: tuple[float, float], offset_at_full: float, with_fsoe: bool
    ) -> list[float]:
    """
    Run the entire pipeline for VB with grouping:
    Grouping -> Aggregation -> Optimization -> Disaggregation
    :param events: Charging events
    :param costs: Price time series
    :param eta: Charging efficiency
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

    # Group
    groups = rs.vb.group(events, est, lst)

    # Optimize
    load = [0] * n_t
    for group in groups:
        p = optimize(
            events=group, costs=costs, eta=eta, delta_t=delta_t,
            support_point=support_point, offset_at_full=offset_at_full, with_fsoe=with_fsoe
        )

        # Disaggregation
        disagg_p = rs.disaggregate(group, p, eta, delta_t, disagg_option)
        for t, val in enumerate(disagg_p):
            load[t] += val
    return load, len(groups)
