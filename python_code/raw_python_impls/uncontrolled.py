"""
Uncontrolled charging.
"""
import python_code.profile_generator as ev

def run(
        *, agents: ev.Agent, eta: float, soc_start: float, delta_t: float
    ) -> dict[int, dict[int, float]]:
    """
    Run uncontrolled charging.
    EVs charge as soon as they arrive at a charging location and with maximum power.
    :param agents: Agents
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :param delta_t: Time step length [hour]
    """
    demands = {agent.id: {} for agent in agents}
    for agent in agents:
        soe = agent.capacity * soc_start # State of energy, Starting condition
        for event in agent.events:
            for t in range(event.start, event.stop):
                # Charge maximum possible
                e_charge = max(0, min(event.p_max * delta_t * eta, agent.capacity - soe))
                soe += e_charge # Charged energy to battery
                if e_charge > 0:
                    demands[agent.id][t] = e_charge / (delta_t * eta)
            soe -= event.consumption # Consumption between charging events
            if soe < 0: # Infeasible -> Slack
                soe = 0
    return demands
