"""
DFO implementation
based on: https://doi.org/10.1109/TKDE.2015.2445755
"""
from dataclasses import dataclass
import numpy as np
import python_code.profile_generator as ev
from . import fo_grouping as fogrp

@dataclass
class ProfileSlice:
    """
    Slice of the FO profile
    """
    amin: float
    amax: float

@dataclass
class FlexObject:
    """
    Flex object
    """
    t_earliest: int
    t_latest: int
    ammount_profile: list[ProfileSlice]
    # The "instance" of the flex object
    t_realized: int
    profile_realized: list[float]

def aggregate_n_1(fos: list[FlexObject]) -> FlexObject:
    """
    Aggregate N FlexObjects to 1. With start alignment.
    :param fos: FlexObjects
    """
    # Determine earliest and latest start time
    t_earliest = min(fos, key=lambda x: x.t_earliest).t_earliest
    min_diff_obj = min(fos, key=lambda x: x.t_latest - x.t_earliest)
    t_latest =  t_earliest + (min_diff_obj.t_latest - min_diff_obj.t_earliest)

    # Create profile
    latest_time_object =  max(fos, key=lambda x: x.t_earliest + len(x.ammount_profile))
    latest_time = latest_time_object.t_earliest + len(latest_time_object.ammount_profile)
    ammount_profile = []
    for _ in range(latest_time - t_earliest):
        ammount_profile.append( ProfileSlice(0, 0) )

    for o in fos:
        offeset = o.t_earliest - t_earliest
        for t, s in enumerate(o.ammount_profile):
            ammount_profile[t+offeset].amin += s.amin
            ammount_profile[t+offeset].amax += s.amax

    return FlexObject(t_earliest, t_latest, ammount_profile, None, None)

def aggregate_n_m(fos, est, tft):
    """
    Aggregate N FlexObjects to M.
    :param fos: FlexObjects
    :param est: earliest start time tolerance
    :param tft: time flexibility tolerance
    """
    groups = fogrp.pre_group(
        fos, est, tft, lambda x: x.t_earliest, lambda x: x.t_latest - x.t_earliest
    )
    opt_groups = fogrp.optimize_groups(
        groups, est, tft, lambda x: x.t_earliest, lambda x: x.t_latest - x.t_earliest
    )
    agg_objects = []
    for group in opt_groups:
        agg_objects.append(
            aggregate_n_1([o for c in group.cells for o in c.objects])
        )
    return agg_objects, opt_groups

def disaggregate_1_n(fos: list[FlexObject], fo_a: FlexObject, alignments: list[int]):
    """
    Disaggreagte 1 FlexObjects to N.
    :param fos: FlexObjects
    :param fo_a: Aggregated FlexObject
    :param alignments: Alignmnets of the FlexObjects (est for start aligned objects)
    """
    relative_profile = []
    for t, _ in enumerate(fo_a.profile_realized):
        s = fo_a.ammount_profile[t]
        sx = fo_a.profile_realized[t]
        if s.amax == s.amin:
            relative_profile.append( 1 )
        else:
            relative_profile.append( (sx - s.amin) / (s.amax - s.amin) )

    for i, o in enumerate(fos):
        o.t_realized = fo_a.t_realized - fo_a.t_earliest + alignments[i]
        profile_realized = []
        for j, s in enumerate(o.ammount_profile):
            profile_realized.append(
                 s.amin + (s.amax - s.amin) * relative_profile[alignments[i] - fo_a.t_earliest + j]
            )
        o.profile_realized = profile_realized
    return fos

def flex_objects_from_agents(
        agents: list[ev.Agent], eta: float, soc_start: float, delta_t: float
    ) -> list[FlexObject]:
    """
    Create flex objects form agents.
    :param agents: Agents
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :param delta_t: Time step length [hour]
    """
    fos = []
    for agent in agents:
        soe = agent.capacity * soc_start # State of energy, Starting condition

        for event in agent.events:
            ammount_profile = []
            charged_total = 0
            for _ in range(event.start, event.stop):
                # Charge maximum possible
                e_charge = max(0, min(event.p_max * eta * delta_t, agent.capacity - soe))
                soe += e_charge  # Charged energy to battery
                if e_charge > 0:
                    charged_total += e_charge
                    ammount_profile.append(ProfileSlice(e_charge / eta, e_charge / eta))
                else:
                    break

            soe -= event.consumption # Consumption between charging events

            # Slack
            if soe < 0:
                soe = 0

            if charged_total > 0:
                fos.append(
                    FlexObject(
                        event.start,
                        event.stop - len(ammount_profile),
                        ammount_profile,
                        None,
                        None
                    )
                )
    return fos

def optimzie_flex_objects(fos: list[FlexObject], costs: list[float]):
    """
    Optimize flex objects. There is probably a faster way.
    But the time cost of this method is currently irrelevant compared to grouping.
    :param fos: FlexObjects. Results are stored in .profile_realized
    :param costs: Price time series
    """
    for o in fos:
        z_best = np.inf
        best_start = None
        for t_start in range(o.t_earliest, o.t_latest + 1):
            z = 0
            for i, s in enumerate(o.ammount_profile):
                z += s.amin * costs[t_start+i]
            if z < z_best:
                z_best = z
                best_start = t_start
        o.t_realized = best_start
        o.profile_realized = [s.amin for s in o.ammount_profile]

def pipeline(
        agents: list[ev.Agent], eta: float, soc_start: float, delta_t: float,
        costs: list[float], est: float, tft: float
    ) -> tuple[list[float], int]:
    """
    Run the entire pipeline of FOs
    Create FOs -> Aggregation -> Optimization -> Disaggregation
    :param agents: Agents
    :param eta: Charging efficiency
    :param soc_start: SOC at start
    :param delta_t: Time step length [hour]
    :param costs: Cost time series
    :param est: earliest start time tolerance
    :param lst: latest stop time tolerance
    """
    fos = flex_objects_from_agents(agents, eta, soc_start, delta_t)
    agg_objects, groups = aggregate_n_m(fos, est, tft)
    optimzie_flex_objects(agg_objects, costs)

    # Disaggregate
    instances = []
    for fo_a, group in zip(agg_objects, groups):
        fos = [o for c in group.cells for o in c.objects]
        instances.extend( disaggregate_1_n(fos, fo_a, [o.t_earliest for o in fos]) )

    # Get dispatched power
    rslt = [0] * len(costs)
    for o in instances:
        for i, p in enumerate(o.profile_realized):
            rslt[i+o.t_realized] += p/delta_t
    return rslt, len(agg_objects)
