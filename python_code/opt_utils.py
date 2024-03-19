"""
Utility for optimization. Used for optimal smart charging and virtual battery. 
"""
from typing import NamedTuple
import python_code.profile_generator as ev

class ChrgEvent(NamedTuple):
    """
    Charging event
    """
    p_max: float
    start: int
    stop: int
    e_arrival: float
    e_departure: float
    capacity: float
    consumption: float
    slack: float

    def calc_e_bat_max(self, eta: float, delta_t: float):
        """
        Determine the maximum battery level at every time step.
        :param eta: Charging efficiency
        :param delta_t: Time step length [h]
        """
        e_max = {}
        soe = self.e_arrival
        for t in range(self.start, self.stop):
            # Charge maximum possible
            e_charge = max(0, min(self.p_max * eta * delta_t, self.capacity - soe))
            soe += e_charge
            e_max[t+1] = soe
        return e_max

    def calc_e_bat_min(self, eta: float, delta_t: float):
        """
        Determine the minimum battery level at every time step.
        :param eta: Charging efficiency
        :param delta_t: Time step length [h]
        """
        e_min = {}
        e_bat_reversed = self.e_departure
        for t in range(self.stop, self.start, -1):
            e_min[t] = e_bat_reversed
            e_bat_reversed -= max(0, min(self.p_max * eta * delta_t, e_bat_reversed-self.e_arrival))
            if e_bat_reversed < 0: # eBat can't go below zero
                e_bat_reversed = 0
        return e_min


def get_slack(
        *, agent: ev.Agent, events: list[ev.Event], e_bat_start: float,
        eta: float, delta_t: float
    ) -> (dict[int, float], float):
    """
    Determine how much of the energy consumption is infeasible even then
    charging as much as possible.
    :param agent: Agent
    :param events: Parking events of agent
    :param e_bat_start: Energy content at start
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :return: dict[Index of event, consumption in following driving event that is infeasible]
    """
    slacks = {}
    soe = e_bat_start # State of charge, Starting condition
    for eventidx, event in enumerate(events):
        e_charge = max( # Charge maximum possible
            0,
            min(event.p_max * eta * delta_t * (event.stop-event.start), agent.capacity - soe)
        )
        soe += e_charge # Charged energy to battery
        soe -= event.consumption # Consumption between charging events
        if soe < 0: # Infeasible -> Slack
            slack = abs(soe)
            soe = 0
        else:
            slack = 0
        slacks[eventidx] = slack
    max_ebat_end = soe
    return slacks, max_ebat_end

def get_chrg_events_restricted(
        *, agent: ev.Agent, eta: float, soc_start: float, delta_t: float
    ) -> list[ChrgEvent]:
    """
    Determine charging events.
    e_departure will be the maximum possible energy content at departure.
    :param agent: Agent
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :pram delta_t: Time step length [hour]
    """
    chrg_events = []
    soe = agent.capacity * soc_start # State of energy, Starting condition
    for event in agent.events:
        e_arrival = soe

        # Charge maximum possible
        e_charge = max(0, min(event.p_max * eta * delta_t * (event.stop - event.start), agent.capacity - soe))
        soe += e_charge
        e_departure = soe

        # Consumption between charging events
        soe -= event.consumption

        # Slack
        if soe < 0:
            slack = abs(soe)
            soe = 0
        else:
            slack = 0

        chrg_events.append(
            ChrgEvent(
                p_max=event.p_max, start=event.start,
                stop=event.stop, e_arrival=e_arrival, e_departure=e_departure,
                capacity=agent.capacity, consumption=event.consumption, slack=slack
            )
        )
    return chrg_events
