"""
DFO implementation (rust)
based on: https://doi.org/10.1145/2934328.2934339
"""

# pylint: disable=import-error
import ev_agg_bench_rs as rs # type: ignore
# pylint: enable=import-error

import gurobipy as grb
# pylint fails to look into gurobipy
# pylint: disable=no-member

def optimize(dfo: rs.dfo.DFO, costs: list[float], eps: float = 1e-5) -> list[float]:
    """
    Optimal dispatch for DFO
    :param dfo: DFO
    :param costs: Cost time series
    :param eps: Geometric percision
    """
    n_t = len(dfo.slices)
    cost_slice = costs[dfo.start: dfo.stop]

    env_params = {"OutputFlag": 0, "FeasibilityTol": eps}
    with grb.Env(params=env_params) as env, grb.Model(env=env) as model:
        # Vars
        e = {t: model.addVar() for t in range(n_t)}
        d = {t: grb.quicksum([e[tt] for tt in range(t)]) for t in range(n_t)}

        # Constraints
        for t, s in enumerate(dfo.slices):
            s = s.vertices
            hull = rs.dfo.gift_wrapping([(x.d, x.e) for x in s], eps)

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

def pipeline(*, events, eta, delta_t, costs, num_samples, est, lst, eps):
    """
    Run the entire pipeline of DFOs (RUST Implementation):
    Create DFOs -> Aggregation -> Optimization -> Disaggregation
    :param events: Charging events
    :param eta: Charging efficiency
    :param delta_t: Time step length [hour]
    :param costs: Cost time series
    :param num_samples: Number of paths for dfo estimation. More paths -> more percise approximation
    :param est: earliest start time tolerance
    :param lst: latest stop time tolerance
    :param eps: Geometric percision
    """

    dfos, groups = rs.dfo.aggregate_pipeline(events, eta, delta_t, num_samples, eps, est, lst)

    ys = []
    for dfo_a in dfos:
        y = optimize(dfo_a, costs, eps=eps)
        ys.append(y)

    rslt = rs.dfo.disaggregate_and_add(dfos, ys, groups, eta, delta_t, eps)
    return rslt, len(groups)
