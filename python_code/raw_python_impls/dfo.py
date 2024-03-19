"""
DFO implementation
based on: https://doi.org/10.1145/2934328.2934339
"""
from dataclasses import dataclass

import numpy as np
import gurobipy as grb

import python_code.profile_generator as ev
from . import fo_grouping as fogrp

# pylint fails to look into gurobipy
# pylint: disable=no-member

@dataclass
class DFOVertex:
    """
    Vertex of a DFO slice
    """
    d: float
    e: float

@dataclass
class ChrgFlexModel:
    """
    Flexibility model for a charging event
    """
    start: int
    stop: int
    p_max: float
    energy_needed: float

@dataclass
class DFO:
    """
    DFO
    """
    t_start: int
    t_stop: int
    # Slices. Vertices must form a point, straight line, or convex polygon.
    # If the vertices form a polygon they must form a linear ring.
    # The first value is NOT repeated at the end.
    polyhs: list[list[DFOVertex]]
    original_data: object # For testing

    def reduce(self, eps: float):
        """
        Find a more compact representation of the slices
        :param eps: Geometric percision.
        """
        n_t = len(self.polyhs)
        for i in range(n_t):
            vertices = self.polyhs[i]
            if len(vertices) <= 3:
                continue

            endpoint = vertices[0]
            canditate = vertices[1]
            new_vertices = [endpoint]
            for test_point in vertices[2:] + [endpoint]:
                a = endpoint.d - canditate.d
                b = test_point.e - canditate.e
                c = endpoint.e - canditate.e
                d = test_point.d - canditate.d
                v = a * b - c * d
                # Are the three points not on a line?
                if abs(v) >= eps:
                    endpoint = canditate
                    new_vertices.append(endpoint)
                # Does the line turn by 180° at the canditate?
                elif (a * d >= 0) and (b * c >= 0):
                    endpoint = canditate
                    new_vertices.append(endpoint)
                canditate = test_point
            self.polyhs[i] = new_vertices

def chrg_flex_models_from_agents(
        agents: list[ev.Agent], eta: float, soc_start: float, delta_t: float
    ) -> list[ChrgFlexModel]:
    """
    Determine flexibility models from agents.
    :param agents: Agents
    :param eta: Charging efficiency
    :param soc_start: Starting SOC [0, 1]
    :param delta_t: Time step length [hour]
    """
    flex_models = []

    for agent in agents:
        soe = agent.capacity * soc_start # State of energy, starting condition

        for event in agent.events:
            e_needed = 0
            for _ in range(event.start, event.stop):
                # Charge maximum possible
                e_charge = max(0, min(event.p_max * eta * delta_t, agent.capacity - soe))
                soe += e_charge  # Charged energy to battery
                if e_charge > 0:
                    e_needed += e_charge
                else:
                    break

            soe -= event.consumption # Consumption between charging events

            # Slack
            if soe < 0:
                soe = 0

            if e_needed > 0:
                flex_models.append(
                    ChrgFlexModel(
                        event.start,
                        event.stop,
                        event.p_max,
                        e_needed
                    )
                )
    return flex_models

def dfo_inner(chrg_flex_model: ChrgFlexModel, num_samples: int, eta: float, delta_t: float) -> DFO:
    """
    Create inner approximation of DFO
    :param chrg_flex_model: Charging event flexibility model
    :param num_samples: Number of paths for dfo estimation. More paths -> more percise approximation
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    """
    n_t = chrg_flex_model.stop - chrg_flex_model.start
    e_needed = chrg_flex_model.energy_needed
    p_max = chrg_flex_model.p_max
    polyhs = []
    d_min, d_max= None, None

    for t in range(n_t):
        poly_max = []
        poly_min = []
        if t == 0:
            v_max = min(p_max * eta * delta_t, e_needed)
            v_min = min(v_max, max(0, e_needed - p_max * eta * delta_t * (n_t-1)))
            poly_max.append( DFOVertex(d=0, e=v_max) )
            poly_min.append( DFOVertex(d=0, e=v_min) )
        else:
            for s in range(num_samples):
                d = d_min + s * (d_max - d_min) / (num_samples - 1)
                energy_left = e_needed - d
                v_max = min(p_max * eta * delta_t, energy_left)
                v_min = min(v_max, max(0, energy_left - p_max * eta * delta_t * (n_t-t-1)))
                poly_max.append( DFOVertex(d=d, e=v_max) )
                poly_min.append( DFOVertex(d=d, e=v_min) )
        min_sum_vertex = min(poly_min, key=lambda x: x.d + x.e)
        d_min = min_sum_vertex.d + min_sum_vertex.e
        max_sum_vertex = max(poly_max, key=lambda x: x.d + x.e)
        d_max = max_sum_vertex.d + max_sum_vertex.e
        polyhs.append(poly_max + poly_min[::-1])
    dfo = DFO(
        t_start=chrg_flex_model.start, t_stop=chrg_flex_model.stop,
        polyhs=polyhs, original_data=chrg_flex_model
    )
    dfo.reduce(1e-5) # Simplify DFO
    return dfo

def get_dep_energy_min_max(vertices: list[DFOVertex]) -> tuple[float, float]:
    """
    Get minimum and maximum dependency ammount
    :param vertices: Vertices of one DFO slice
    """
    dmin = np.inf
    dmax = 0
    for v in vertices:
        if dmin > v.d:
            dmin = v.d
        if dmax < v.d:
            dmax = v.d
    return dmin, dmax

def get_total_en_min_max(vertices: list[DFOVertex]) -> tuple[float, float]:
    """
    Get minimum and maximum total energy
    :param vertices: Vertices of one DFO slice
    """
    emin = np.inf
    emax = 0
    for v in vertices:
        val = v.e + v.d
        if emin > val:
            emin = val
        if emax < val:
            emax = val
    return emin, emax

def get_energy_min_max(vertices: list[DFOVertex], d: float, eps:float = 1e-6) -> tuple[float, float]:
    """
    Get minimum and maximum energy
    :param vertices: Vertices of one DFO slice
    :param d: Dependency ammount.
    :param eps: Geometric percision.
    """
    e_min = np.inf
    e_max = 0
    last = vertices[0]
    found = False
    for v in vertices[1:] + [vertices[0]]:
        # d is exactly at an existing vertice
        if d+eps >= v.d >= d-eps:
            found = True
            if e_min > v.e:
                e_min = v.e
            if e_max < v.e:
                e_max = v.e
        else:
            if v.d >= last.d:
                left = last
                right = v
            else:
                left = v
                right = last
            # Is d between the to points?
            if left.d <= d <= right.d:
                found = True
                # Interpolation of e value at d
                e = left.e + (d - left.d) * (right.e - left.e) / (right.d - left.d)
                if e_min > e:
                    e_min = e
                if e_max < e:
                    e_max = e
        last = v
    # Is d inside the polygon?
    if not found:
        raise AssertionError("d is not inside polygon")
    return e_min, e_max

def add_slices_infront(dfo: DFO, n: int) -> DFO:
    """
    Add empty slices at the start of DFO
    :param dfo: DFO
    :param n: Number of slices
    """
    for _ in range(n):
        dfo.polyhs = [[DFOVertex(0, 0)]] + dfo.polyhs
    dfo.t_start -= n
    return dfo

def add_slices_end(dfo: DFO, n: int) -> DFO:
    """
    Add empty slices at the end of DFO
    :param dfo: DFO
    :param n: Number of slices
    """
    s_min, s_max = get_total_en_min_max(dfo.polyhs[-1])
    for _ in range(n):
        dfo.polyhs = dfo.polyhs + [[DFOVertex(s_min, 0), DFOVertex(s_max, 0)]]
    dfo.t_stop += n
    return dfo

def prepare_for_agg(dfo1: DFO, dfo2: DFO) -> tuple[DFO, DFO]:
    """
    Prepare two DFOs for aggregation by adding empty
    slice if start and end times don't match.
    :param dfo1: First DFO
    :param dfo2: Second DFO
    """
    # Add slices to front if needed
    if dfo1.t_start > dfo2.t_start:
        dfo1 = add_slices_infront(dfo1, dfo1.t_start - dfo2.t_start)
    elif dfo2.t_start > dfo1.t_start:
        dfo2 = add_slices_infront(dfo2, dfo2.t_start - dfo1.t_start)
    # Add slices to end if needed
    if dfo1.t_stop < dfo2.t_stop:
        dfo1 = add_slices_end(dfo1, dfo2.t_stop - dfo1.t_stop)
    elif dfo2.t_stop < dfo1.t_stop:
        dfo2 = add_slices_end(dfo2, dfo1.t_stop - dfo2.t_stop)
    return dfo1, dfo2

def agg2to1(dfo1: DFO, dfo2: DFO, num_samples: int, eps: float = 1e-5) -> DFO :
    """
    Aggregate 2 DFOs to 1
    :param dfo1: First DFO
    :param dfo2: Second DFO
    :param num_samples: Number of paths for dfo estimation. More paths -> more percise approximation
    :param eps: Geometric percision.
    """
    dfo1, dfo2 = prepare_for_agg(dfo1, dfo2)

    da_min = 0
    da_max = 0
    polyhs = []
    eta1 = 1
    eta2 = 1

    n_t = len(dfo1.polyhs)

    for t in range(n_t):
        d1_min, d1_max = get_dep_energy_min_max(dfo1.polyhs[t])
        d2_min, d2_max = get_dep_energy_min_max(dfo2.polyhs[t])
        s1_min, s1_max = get_total_en_min_max(dfo1.polyhs[t])
        s2_min, s2_max = get_total_en_min_max(dfo2.polyhs[t])

        poly_max = []
        poly_min = []
        for k in range(num_samples):
            da = da_min + k * (da_max - da_min) / (num_samples - 1)
            d1 = d1_min + k * (d1_max - d1_min) / (num_samples - 1)
            d2 = d2_min + k * (d2_max - d2_min) / (num_samples - 1)
            e1_min, e1_max = get_energy_min_max(dfo1.polyhs[t], d1, eps)
            e2_min, e2_max = get_energy_min_max(dfo2.polyhs[t], d2, eps)

            if t < n_t - 1:
                # Handle dummy slices
                if ((s2_max - s2_min) <= 1e-9) or ((s1_max - s1_min)  <= 1e-9):
                    s1_upper = d1 + e1_max
                    s2_upper = d2 + e2_max
                    s1_lower = d1 + e1_min
                    s2_lower = d2 + e2_min
                # Normal case
                else:
                    s1_range = s1_max - s1_min
                    s2_range = s2_max - s2_min

                    # Upper bound s
                    s1_upper_option1 = d1 + e1_max
                    s1_upper_option2 = (d2 + e2_max - s2_min) / s2_range * s1_range + s1_min
                    s1_upper = min(s1_upper_option1, s1_upper_option2)
                    s2_upper = (s1_upper - s1_min) / s1_range * s2_range + s2_min

                    # Lower bound s
                    s1_lower_option1 = d1 + e1_min
                    s1_lower_option2 = (d2 + e2_min - s2_min) / s2_range * s1_range + s1_min
                    s1_lower = max(s1_lower_option1, s1_lower_option2)
                    s2_lower = (s1_lower - s1_min) / s1_range * s2_range + s2_min

                ea_max = s1_upper + s2_upper - da
                ea_min = s1_lower + s2_lower - da

                if (e1_max - e1_min) != 0:
                    eta1 = eta1  * (s1_upper - s1_lower) / (e1_max - e1_min)
                if (e2_max - e2_min) != 0:
                    eta2 = eta2  * (s2_upper - s2_lower) / (e2_max - e2_min)
            else:
                ea_min = e1_min + e2_min
                ea_max = e1_max + e2_max
            poly_max.append( DFOVertex(d=da, e=ea_max) )
            poly_min.append( DFOVertex(d=da, e=ea_min) )
        poly = poly_max + poly_min[::-1]
        polyhs.append(poly)
        da_min, da_max = get_total_en_min_max(poly)
    dfo = DFO(dfo1.t_start, dfo1.t_stop, polyhs, original_data=None)
    dfo.reduce(1e-5) # Simplify DFO
    return dfo, eta1, eta2

def aggregate_n_m(dfos: list[DFO], est: int, lst: int, num_samples: int):
    """
    Aggregate N DFOs to M
    :param dfos: DFOs to aggregate
    :param est: earliest start time tolerance
    :param lst: latest stop time tolerance
    :param num_samples: Number of paths for dfo estimation. More paths -> more percise approximation
    """
    groups = fogrp.pre_group(dfos, est, lst, lambda x: x.t_start, lambda x: x.t_stop)
    opt_groups = fogrp.optimize_groups(groups, est, lst, lambda x: x.t_start, lambda x: x.t_stop)
    agg_objects = []
    for group in opt_groups:
        grp_dfos = [o for c in group.cells for o in c.objects]
        dfo_a = grp_dfos[0]
        for d in grp_dfos[1:]:
            dfo_a, _, _ = agg2to1(d, dfo_a, num_samples)
        agg_objects.append( dfo_a )
    return agg_objects, opt_groups

def disagg1to2(dfo1: DFO, dfo2: DFO, dfo_a: DFO, y: list[float], eps: float = 1e-5):
    """
    Disaggregate 1 DFO to 2
    :param dfo1: First DFO
    :param dfo2: Second DFO
    :param dfo_a: Aggregated DFO
    :param y: Realized power of the aggregated DFO
    :param eps: Geometric percision.
    """
    sa = 0
    s1 = 0
    s2 = 0

    y1 = []
    y2 = []

    n_t = len(dfo_a.polyhs)

    for t in range(n_t):
        if t < n_t - 1:
            sa += y[t]
            sa_min, sa_max = get_total_en_min_max(dfo_a.polyhs[t],)
            f = 0
            if (sa_max - sa_min) != 0:
                f = (sa - sa_min) / (sa_max - sa_min)

            s1_min, s1_max = get_total_en_min_max(dfo1.polyhs[t])
            y1.append(s1_min + f * (s1_max - s1_min) - s1)
            s1 += y1[-1]

            s2_min, s2_max = get_total_en_min_max(dfo2.polyhs[t])
            y2.append(s2_min + f * (s2_max - s2_min) - s2)
            s2 += y2[-1]
        else:
            ea_min, ea_max = get_energy_min_max(dfo_a.polyhs[t], sa, eps=eps)
            f = 0
            if (ea_max - ea_min) != 0:
                f = (y[t] - ea_min) / (ea_max - ea_min)

            e1_min, e1_max = get_energy_min_max(dfo1.polyhs[t], s1, eps=eps)
            y1.append(e1_min + f * (e1_max - e1_min))

            e2_min, e2_max = get_energy_min_max(dfo2.polyhs[t], s2, eps=eps)
            y2.append(e2_min + f * (e2_max - e2_min))
    return y1, y2

def disaggregate_1_n(
        dfo_a: DFO, y: list[float], dfos: list[DFO], eps: float = 1e-5
    ) -> list[list[float]]:
    """
    Disaggregate 1 DFO to N dfos
    :param dfo_a: Aggregated DFO
    :param y: Realized power of the aggregated DFO
    :param dfos: List of DFOs that dfo_a was aggregated from
    :param eps: Geometric percision
    """
    ys = []
    dummy_dfo = dfos[0]
    for dfo in dfos:
        dfo, _ = prepare_for_agg(dfo, dfo_a)
        _, y2 = disagg1to2(dummy_dfo, dfo, dfo_a, y, eps=eps)
        ys.append(y2)
    return ys

def gift_wrapping(pnts: list[(float, float)], eps: float = 1e-5) -> list[(float, float)]:
    """
    Get convex hull of multiple points
    :param pnts: Points
    :param eps: Geometric percision
    """
    convex_hull = []
    point_on_hull = min(pnts, key=lambda x: x[0])

    for _ in range(len(pnts)):
        convex_hull.append(point_on_hull)
        endpoint = pnts[0]

        for canditate in pnts[1:]:
            if endpoint == point_on_hull:
                endpoint = canditate
                continue

            # Line check
            v = (
                (endpoint[0] - point_on_hull[0]) * (canditate[1] - point_on_hull[1]) -
                (endpoint[1] - point_on_hull[1]) * (canditate[0] - point_on_hull[0])
            )

            if v > eps:
                endpoint = canditate
            elif v > -eps:
                # Compute distances
                sqrd_endpnt = (
                    (endpoint[0] - point_on_hull[0])**2 +
                    (endpoint[1] - point_on_hull[1])**2
                )
                sqrd_canditate = (
                    (canditate[0] - point_on_hull[0])**2 +
                    (canditate[1] - point_on_hull[1])**2
                )

                # Check if candiate and endpoint are identical to avoid infinite loops
                if abs(sqrd_canditate - sqrd_endpnt) < eps:
                    if canditate == convex_hull[0]:
                        endpoint = canditate
                # Else keep the furthest point from point_on_hull
                elif sqrd_canditate > sqrd_endpnt:
                    endpoint = canditate
        point_on_hull = endpoint

        if point_on_hull == convex_hull[0]:
            break
    return convex_hull

def optimize(dfo: DFO, costs: list[float], eps: float = 1e-5) -> list[float]:
    """
    Optimal dispatch for DFO
    :param dfo: DFO
    :param costs: Cost time series
    :param eps: Geometric percision
    """
    n_t = len(dfo.polyhs)
    cost_slice = costs[dfo.t_start: dfo.t_stop]

    env_params = {"OutputFlag": 0, "FeasibilityTol": eps}
    with grb.Env(params=env_params) as env, grb.Model(env=env) as model:
        # Vars
        e = {t: model.addVar() for t in range(n_t)}
        d = {t: grb.quicksum([e[tt] for tt in range(t)]) for t in range(n_t)}

        # Constraints
        for t, s in enumerate(dfo.polyhs):
            hull = gift_wrapping([(x.d, x.e) for x in s], eps)

            # No feasible region
            if len(hull) == 0:
                raise AssertionError(f"Slice {t} has no feasible region!")
            # Feasible region is a point
            elif len(hull) == 1:
                model.addLConstr(lhs = e[t], sense = grb.GRB.EQUAL, rhs = hull[0][1])
                model.addLConstr(lhs = d[t], sense = grb.GRB.EQUAL, rhs = hull[0][0])
            # Feasible region is a line
            elif len(hull) == 2:
                pnt_a = hull[0]
                pnt_b = hull[1]
                v = (
                    (pnt_b[0] - pnt_a[0]) * (e[t] - pnt_a[1]) -
                    (pnt_b[1] - pnt_a[1]) * (d[t] - pnt_a[0])
                )
                model.addLConstr(lhs = v,
                                 sense = grb.GRB.GREATER_EQUAL,
                                 rhs = -eps)
                model.addLConstr(lhs = v,
                                 sense = grb.GRB.LESS_EQUAL,
                                 rhs = eps)
                model.addLConstr(lhs = e[t],
                                 sense = grb.GRB.GREATER_EQUAL,
                                 rhs = min(hull, key=lambda x: x[1])[1])
                model.addLConstr(lhs = e[t],
                                 sense = grb.GRB.LESS_EQUAL,
                                 rhs = max(hull, key=lambda x: x[1])[1])
                model.addLConstr(lhs = d[t],
                                 sense = grb.GRB.GREATER_EQUAL,
                                 rhs = min(hull, key=lambda x: x[0])[0])
                model.addLConstr(lhs = d[t],
                                 sense = grb.GRB.LESS_EQUAL,
                                 rhs = max(hull, key=lambda x: x[0])[0])
            # Feasible region is a polygon
            else:
                for j, pnt_a in enumerate(hull):
                    if j == (len(hull) - 1):
                        pnt_b = hull[0]
                    else:
                        pnt_b = hull[j+1]

                    # Check sense
                    if j == 0:
                        pnt_c = hull[-1]
                    else:
                        pnt_c = hull[j-1]
                    v_test = (
                        (pnt_b[0] - pnt_a[0]) * (pnt_c[1] - pnt_a[1]) -
                        (pnt_b[1] - pnt_a[1]) * (pnt_c[0] - pnt_a[0])
                    )
                    sense = grb.GRB.GREATER_EQUAL if v_test >= 0 else grb.GRB.LESS_EQUAL

                    v = (
                        (pnt_b[0] - pnt_a[0]) * (e[t] - pnt_a[1]) -
                        (pnt_b[1] - pnt_a[1]) * (d[t] - pnt_a[0])
                    )
                    model.addLConstr(lhs = v, sense = sense, rhs = 0)

        # Objective
        objective = grb.LinExpr()
        for t in range(n_t):
            objective.addTerms(cost_slice[t], e[t])
        model.ModelSense = grb.GRB.MINIMIZE
        model.setObjective(objective)

        # Solve
        model.optimize()

        # Check Success
        if model.status == 13:
            print("Solution suboptimal!")
        elif model.status > 2:
            raise AssertionError(f"Model terminated! Status {model.status}")

        y = [e[t].X for t in range(n_t)]
    return y

def check_violations(
        dfos: list[DFO], ys: list[list[float]], eta: float, delta_t: float, warn_n: int, eps: float=1e-5
    ) -> bool:
    """
    Check if DFO aggregation and disaggregation result is feasible
    :param dfos: DFOs
    :param ys: Realized power of the DFOs
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :param eps: Precision
    :param warn_n: Warn if more than n violations occur
    """
    total_energy_errors = 0
    to_early_errors = 0
    to_late_errors = 0
    max_power_errors = 0
    for dfo, y in zip(dfos, ys):
        if abs(sum(y) - dfo.original_data.energy_needed) > eps:
            total_energy_errors += 1
        for j, v in enumerate(y):
            if (v > eps) and ((j + dfo.t_start) < dfo.original_data.start):
                to_early_errors += 1
            if (v > eps) and ((j + dfo.t_start) > dfo.original_data.stop):
                to_late_errors += 1
            if v > dfo.original_data.p_max*eta*delta_t + eps:
                max_power_errors += 1
    if total_energy_errors + to_early_errors + to_late_errors + max_power_errors == 0:
        return False
    else:
        if total_energy_errors + to_late_errors + max_power_errors > warn_n:
            print(f"Total energy violations: {total_energy_errors}")
            print(f"Time window violations: {to_early_errors+to_late_errors}")
            print(f"Maximum power violations: {max_power_errors}")
        return True

def pipeline(
        agents: list[ev.Agent], eta: float, delta_t: float, costs: list[float],
        soc_start: float, warn_n: int, num_samples: int = 4, est: int = 50,
        lst: int = 50, eps: float = 1e-5, check: bool = False) -> tuple[list[float], int]:
    """
    Run the entire pipeline of DFOs:
    Create DFOs -> Aggregation -> Optimization -> Disaggregation
    :param agents: Agents
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :param costs: Cost time series
    :param soc_start: SOC at start
    :param warn_n: Warn if more than n violations occur in disaggregation
    :param num_samples: Number of paths for dfo estimation. More paths -> more percise approximation
    :param est: earliest start time tolerance
    :param lst: latest stop time tolerance
    :param eps: Geometric percision
    """
    fmodels = chrg_flex_models_from_agents(agents, eta, soc_start, delta_t)
    dfos = [dfo_inner(mdl, num_samples, eta, delta_t) for mdl in fmodels]
    agg_objects, groups = aggregate_n_m(dfos, est, lst, num_samples)
    rslt = [0] * len(costs)

    for dfo_a, group in zip(agg_objects, groups):
        dfos = [o for c in group.cells for o in c.objects]

        # Optimize
        y = optimize(dfo_a, costs, eps=eps)

        # Disaggregate
        ys = disaggregate_1_n(dfo_a, y, dfos, eps=eps)
        if check:
            _ = check_violations(dfos, ys, eta, delta_t, warn_n, eps=eps)

        # Add to result
        for y, o in zip(ys, dfos):
            for i, p in enumerate(y):
                rslt[i+o.t_start] += p/delta_t/eta
    return rslt, len(agg_objects)
