"""
Sample charging events from MID
"""
from typing import NamedTuple, List
from enum import Enum
from dataclasses import dataclass
import random
import pandas as pd
# pylint: disable=import-error
import ev_agg_bench_rs as rs # type: ignore
# pylint: enable=import-error

class Activity(Enum):
    """
    Activity
    """
    HOME = "home"
    WORK = "work"
    SCHOOL = "school"
    SHOPPING = "shopping"
    OTHER = "other"

class Location(Enum):
    """
    Charging point location
    """
    HOME = "home"
    WORK = "work"
    PUBLIC = "public"

class CarType(Enum):
    """
    Car type [capacity in kwh, CITY consumption in kwh/km]
    based on: https://doi.org/10.1016/j.apenergy.2022.118945
    """
    BEV_SMALL   = [38, 0.146]
    BEV_MEDIUM  = [72, 0.145]
    BEV_LARGE   = [82, 0.182]
    PHEV_MEDIUM = [11.7, 0.24]
    PHEV_LARGE  = [13.9, 0.279]

class Event(NamedTuple):
    """
    Charging event
    """
    start: int
    stop: int
    consumption: float # On the trip after the charging event
    location: Location
    p_max: float

@dataclass
class Agent:
    """
    Electric vehicle agent
    """
    id: int
    events: List[Event]
    capacity: float
    car_type: CarType
    p_home: float
    p_work: float

@dataclass
class Trip:
    """
    Trip
    """
    arrival: int
    departure: int
    activity_at_destination: Activity
    distance_km: float

# Map Activity to a charging location
MAP_PURPOSE_LOCATION ={
    Activity.HOME: Location.HOME, Activity.WORK: Location.WORK,
    Activity.SCHOOL: Location.PUBLIC, Activity.SHOPPING: Location.PUBLIC,
    Activity.OTHER: Location.PUBLIC
}

def get_p_max(agent, location):
    """
    Get the maximum charging rate
    :param agent: Agent
    :param location: Chargin location
    """
    if location == Location.HOME:
        return agent.p_home
    elif location == Location.WORK:
        return agent.p_work
    else:
        return 0

def add_od_activities(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the activities at the origin and destination of each trip
    :param trips: Trip dataframe from MID
    """
    trips = trips.sort_values(["HP_ID_Lok", "W_ID"])  # Sort MID

    sos = trips.W_SO1.values # Activity at the beginning of the day
    wids = trips.W_ID.values
    purposes = trips.zweck.values

    def decode_purpose(j):
        if purposes[j] in [1]:
            return Activity.WORK
        elif purposes[j] in [3]:
            return Activity.SCHOOL
        elif purposes[j] in [4]:
            return Activity.SHOPPING
        elif purposes[j] in [2, 5, 6, 7, 10]:
            return Activity.OTHER
        elif purposes[j] in [8]:
            return Activity.HOME
        elif purposes[j] in [9]: # 9: Return trip -> Next destination is the previous one.
            if wids[j] == 1:
                return None
            else:
                return decode_purpose(j-1)
        else:
            return None

    activity_o_lst = []
    activity_d_lst = []
    for i, so in enumerate(sos):
        activity_d_lst.append(decode_purpose(i))

        activity_o = None
        if wids[i] == 1:
            if so == 1:
                activity_o = Activity.HOME
            elif so in [2]:
                activity_o = Activity.OTHER
        else:
            activity_o = activity_d_lst[i-1]
        activity_o_lst.append(activity_o)

    trips["ActivityAtStart"] = activity_o_lst
    trips["ActivityAtStop"] = activity_d_lst
    return trips

def get_clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up MID data
    :param trips: Trip dataframe from MID
    """
    # Add activity information
    trips = add_od_activities(trips)

    # Drop days with missing or faulty data
    mask = (
        (trips.W_SZS > 25) | (trips.W_SZM > 80) | (trips.W_AZS > 25) | (trips.W_AZM > 80) |
        (trips.W_FOLGETAG > 3) | (trips.ActivityAtStart.isna()) | (trips.ActivityAtStop.isna()) |
        ((trips.W_AZS*60 + trips.W_AZM) < (trips.W_SZS*60 + trips.W_SZM))
    ) & (trips.W_RBW != 1)
    filtered_ids = trips[mask].HP_ID_Lok
    df_clean = trips[(trips.W_RBW == 0) & ~trips.HP_ID_Lok.isin(filtered_ids)].copy()

    # Arrival and departure time
    df_clean["arrival"] = df_clean.W_AZS * 60 + df_clean.W_AZM
    df_clean.loc[df_clean.W_FOLGETAG != 0, "arrival"] = 1440
    df_clean["departure"] = df_clean.W_SZS * 60 + df_clean.W_SZM

    # Trip starts before previous one ended
    df_clean['arrival_prev'] = [None] + list(df_clean["arrival"].values[:-1])
    filtered_ids = df_clean[
        (df_clean.departure < df_clean.arrival_prev) & (df_clean['W_ID'] != 1)
    ].HP_ID_Lok
    df_clean = df_clean[~df_clean.HP_ID_Lok.isin(filtered_ids)]

    # Create proper indexing
    df_clean = df_clean.set_index(['HP_ID_Lok', 'W_ID'])

    # Sort trips
    df_clean = df_clean.sort_index(level=0, sort_remaining=True)
    return df_clean

def get_charging_events(
        trips: pd.DataFrame, agents: list[Agent], ndays: int, seed: int, resolution: int = 15
    ) -> list[Agent]:
    """
    Sample charging events from MID data
    :param trips: Trip dataframe from MID
    :param agents: List of agents for which charging events will be generated.
    :param ndays: Number of days
    :param seed: random seed
    :param resolution: Time resolution in minutes
    """
    rng = random.Random(seed)

    trips = get_clean_trips(trips).reset_index() # Clean up data
    trips = trips[trips.hvm_diff1 == 6] # Only car trips

    # Get trip chains from data
    trips_chains_by_weekday = {i: [] for i in trips.ST_WOTAG.unique() }
    current_chain = []
    current_id = trips["HP_ID_Lok"].values[0]
    current_weekday = trips["ST_WOTAG"].values[0]
    for dep, arr, activity, distance, hpid, weekday in zip(
        trips["departure"], trips["arrival"], trips["ActivityAtStop"],
        trips["wegkm_imp"], trips["HP_ID_Lok"], trips["ST_WOTAG"]):
        if hpid != current_id:
            if len(current_chain) != 0:
                trips_chains_by_weekday[current_weekday].append( current_chain )
            current_chain = []
        current_id = hpid
        current_weekday = weekday
        trip = Trip(
            arrival=arr, departure=dep, distance_km=distance, activity_at_destination=activity
        )
        current_chain.append(trip)

    # Create charging events
    end_int = ndays * 24 * 60 // resolution
    for agent in agents:
        agent_trips = []
        for i in range(ndays):
            weekday = i % 7 + 1
            trip_chain = rng.choice(trips_chains_by_weekday[weekday])
            # Add time offset indicating that the trips are on the ith day
            for trip in trip_chain:
                trip = Trip( # Copy trip
                    trip.arrival, trip.departure, trip.activity_at_destination, trip.distance_km
                )
                offset = i * 1440
                trip.arrival += offset
                trip.departure += offset
                agent_trips.append( trip )
        agent.events.extend(get_chrg_events(agent_trips, agent, end_int, resolution))
    return agents

def get_chrg_events(trips: List[Trip], agent: Agent, end_int: int, resolution: int) -> list[Event]:
    """
    Get charging events.
    :param trips: List of trips
    :param agents: Agent
    :param end_int: Last time step
    :param resolution: Time resolution
    """
    events = []
    saved_consumption = 0
    last_timestep = 0
    current_location = Location.HOME # Start location
    for trip in trips:
        dep = trip.departure // resolution
        arr = trip.arrival // resolution

        # If trip goes ends out of time frame
        if arr >= end_int:
            break

        # Calculate consumption
        consumption = agent.car_type.value[1] * trip.distance_km

        # Parking event happens in an instant -> save consumption and continue
        if last_timestep == dep:
            saved_consumption += consumption
            continue

        event = Event(
            last_timestep, dep, consumption + saved_consumption,
            current_location, get_p_max(agent, current_location)
        )
        events.append(event)

        last_timestep = arr
        current_location = MAP_PURPOSE_LOCATION[trip.activity_at_destination]
        saved_consumption = 0

    # From last event to end of time frame
    event =Event(last_timestep, end_int, 0.0, current_location, get_p_max(agent, current_location))
    events.append(event)
    return events

def get_chrg_events_restricted(
        agents: list[Agent], eta: float, soc_start: float, delta_t: float
    ):
    """
    Determine charging events.
    e_departure will be the maximum possible energy content at departure.
    :param agents: Agents
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :pram delta_t: Time step length [hour]
    """
    chrg_events = []
    for agent in agents:
        soe = agent.capacity * soc_start # State of energy, Starting condition
        for event in agent.events:
            e_arrival = soe

            # Charge maximum possible
            e_charge = max(0, min(event.p_max * eta * delta_t * (event.stop - event.start), agent.capacity - soe))
            soe += e_charge
            e_departure = soe

            # Consumption between charging events
            soe -= event.consumption

            # Slack - Catch impossible driving patterns
            if soe < 0:
                soe = 0

            chrg_events.append(
                rs.ChrgEvent(
                    start=event.start, stop=event.stop, p_max=event.p_max,
                    energy_arrival=e_arrival, energy_departure=e_departure,
                    capacity=agent.capacity
                )
            )
    return chrg_events
