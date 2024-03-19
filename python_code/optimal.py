"""
Optimal smart charging. Optimizes all EVs individually without aggregation.
"""
import gurobipy as grb
import python_code.opt_utils as utils
import python_code.profile_generator as ev

# pylint fails to look into gurobipy
# pylint: disable=no-member

def set_agent_constraints(
        *, model: grb.Model, agent: ev.Agent, events: list[utils.ChrgEvent], max_ebat_end: float,
        e_bat_start: float, eta: float, delta_t: float, event_restricted: bool
    ) -> dict[int, grb.Var]:
    """
    Set the cosntraints for one EV.
    :param model: Gurobi model
    :param agent: Agent
    :param events: Charging events
    :param max_ebat_end: Maximum battery content at the end of the time window
    :param e_bat_start: Energy content at the beginning
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :param event_restricted: Should energy shift be restriceted to individual charging events?
    true -> no shift between events, false -> Energy shift is only restricted by consumption.
    """
    e_bat_pre, e_bat_post, p_charge = {}, {}, {} # Optimization variables
    e_bat_pre[0] = e_bat_start # Starting condition
    for event_idx, event in enumerate(events):
        # Charging power of EV at time step
        for t in range(event.start, event.stop):
            p_charge[t] = model.addVar(ub=event.p_max) # 0 <= p_charge <= P_max[t, car]

        # Battery energy content before event
        e_bat_pre[event_idx + 1] = model.addVar(ub=agent.capacity) # 0 <= eBat <= capacity[car]

        # Battery energy content after event
        if event_restricted:
            e_bat_post[event_idx] = event.e_departure
        else:
            e_bat_post[event_idx] = model.addVar(ub=agent.capacity) # 0 <= eBat <= capacity[car]

        # E_bat after charging = E_bat before charging + charged energy
        charged_energy = grb.quicksum(
            p_charge[t] for t in range(event.start, event.stop)
        ) * eta * delta_t
        model.addLConstr(lhs=charged_energy + e_bat_pre[event_idx],
                         sense=grb.GRB.EQUAL,
                         rhs=e_bat_post[event_idx])

        # E_bat after driving = E_bat before driving - consumption + slack
        model.addLConstr(lhs=e_bat_post[event_idx] - event.consumption + event.slack,
                         sense=grb.GRB.EQUAL,
                         rhs=e_bat_pre[event_idx + 1])
    # Time window energy constraint
    bound_ebat_end = min(max_ebat_end, e_bat_pre[0])
    model.addLConstr(lhs=bound_ebat_end, sense=grb.GRB.LESS_EQUAL, rhs=e_bat_pre[len(events)])
    return p_charge

def optimal(
        *, agents: list[ev.Agent], costs: list[float], eta: float,
        soc_start: float, delta_t: float, event_restricted: bool
    ) -> dict[int, dict[int, float]]:
    """
    Optimal smart charging.
    :param agents: Agents
    :param costs: Price time series
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :param delta_t: Time step length [hour]
    :param event_restricted: Should energy shift be restriceted to individual charging events?
    true -> no shift between events, false -> Energy shift is only restricted by consumption.
    """
    demands = {agent.id: {} for agent in agents}
    with grb.Env(params={"OutputFlag": 0}) as env, grb.Model(env=env) as model:
        p_charge = {}
        for agent in agents:
            e_bat_start = soc_start * agent.capacity # Starting SOC

            # Process data
            _, max_ebat_end = utils.get_slack(
                agent=agent, events=agent.events, e_bat_start=e_bat_start, eta=eta, delta_t=delta_t
            )
            chrg_events = utils.get_chrg_events_restricted(
                agent=agent, eta=eta, soc_start=soc_start, delta_t=delta_t
            )

            # Constraints
            p_charge_agent = set_agent_constraints(
                model=model, agent=agent, events=chrg_events, max_ebat_end=max_ebat_end,
                e_bat_start=e_bat_start, eta=eta, delta_t=delta_t,
                event_restricted=event_restricted)

            # Add pCharge var to set of fleet
            for t, var in p_charge_agent.items():
                p_charge[t, agent.id] = var

        # Objective
        objective = grb.LinExpr()
        for (t, _), var in p_charge.items():
            objective.addTerms(costs[t], var)
        model.ModelSense = grb.GRB.MINIMIZE
        model.setObjective(objective)

        # Calculation
        model.optimize()

        # Solution
        if model.status == 13:
            print("Solution suboptimal!")
        elif model.status > 2:
            raise AssertionError(f"Model terminated! Status: {model.status}")

        # Solution
        for agent in agents:
            for event in agent.events:
                for t in range(event.start, event.stop):
                    p = p_charge[t, agent.id].X
                    if p > 0:
                        demands[agent.id][t] = p
    return demands
