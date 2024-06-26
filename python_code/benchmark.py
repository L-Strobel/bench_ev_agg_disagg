"""
Run benchmark
"""
import timeit
from io import StringIO

import pandas as pd
# pylint: disable=import-error
import ev_agg_bench_rs as rs # type: ignore
# pylint: enable=import-error

import python_code.utils as utils
import python_code.config
import python_code.cost_ts as cts
import python_code.dfo as dfo_rs
import python_code.rep_profile as rep_rs
import python_code.virtual_battery as vb_rs
import python_code.profile_generator as gen

def run_opt(agents, costs, config):
    """
    Run optimal
    """
    t_start = timeit.default_timer()
    opt = utils.get_opt(agents, costs, config)
    t_stop = timeit.default_timer()
    return opt, t_stop - t_start

def run_uncontrolled(agents, costs, config):
    """
    Run uncontrolled
    """
    events = gen.get_chrg_events_restricted(agents, config.ETA, config.SOC_START, config.DELTA_T)
    t_start = timeit.default_timer()
    load = rs.run_unctr(events, config.ETA, config.DELTA_T, len(costs))
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start

def run_rep(agents, costs,  config, seed, n_profiles, disagg_option):
    """
    Run represenative profile
    """
    t_start = timeit.default_timer()
    load = rep_rs.pipeline(agents, seed, n_profiles, disagg_option, config.ETA, config.SOC_START, config.DELTA_T, costs)
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start, n_profiles

def run_fo(agents, costs,  config, est, tf):
    """
    Run FO
    """
    events = gen.get_chrg_events_restricted(agents, config.ETA, config.SOC_START, config.DELTA_T)
    t_start = timeit.default_timer()
    load, n_objects = rs.fo.pipeline(events, config.ETA, config.DELTA_T, costs, est, tf)
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start, n_objects

def run_dfo(agents, costs,  config, est, st, num_samples):
    """
    Run DFO
    """
    events = gen.get_chrg_events_restricted(agents, config.ETA, config.SOC_START, config.DELTA_T)
    t_start = timeit.default_timer()
    try:
        load, n_objects = dfo_rs.pipeline(
            events=events, eta=config.ETA, delta_t=config.DELTA_T, costs=costs, num_samples=num_samples,
            est=est, lst=st, eps=1e-5
        )
        t_stop = timeit.default_timer()
        o_value = utils.calc_total_costs(load, costs)
    except AssertionError:
        t_stop = timeit.default_timer()
        load = None
        n_objects = None
        o_value = None
    return o_value, t_stop - t_start, n_objects

def run_vb(agents, costs, config, disagg_option, mpx, mpy, offset, with_fsoe, grpd, est, st):
    """
    Run virtual battery
    """
    events = gen.get_chrg_events_restricted(agents, config.ETA, config.SOC_START, config.DELTA_T)
    t_start = timeit.default_timer()

    if grpd:
        load, n_objects = vb_rs.pipeline_grpd(
            events=events, eta=config.ETA, delta_t=config.DELTA_T, costs=costs, disagg_option=disagg_option,
            support_point=(mpx, mpy), offset_at_full = offset, with_fsoe = with_fsoe,
            est=est, lst=st
        )
    else:
        load = vb_rs.pipeline(
            events=events, eta=config.ETA, delta_t=config.DELTA_T, costs=costs, disagg_option=disagg_option,
            support_point=(mpx, mpy), offset_at_full = offset, with_fsoe = with_fsoe
        )
        n_objects = 1
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start, n_objects

def run_iteration(trips, n_agents, seed, price_signal, fn_price, fn_gen, fn_dem, config):
    """
    Run benchmark for one seed, agent number, and price signal combination.
    """
    result = []
    res_columns = [
        "Seed", "Method", "Price Signal", "N_Agents",
        "Objective Value", "Gap", "Compression", "Runtime_s",
        "Config"
    ]

    # Prepare
    agents, costs = utils.prepare_run(trips, seed, n_agents, price_signal, fn_price, fn_gen, fn_dem, config)
    n_events = utils.get_event_count(agents, config)

    def write_result_line(name, o_value, time, n_objects):
        if (o_value is None) or (opt is None):
            gap = None
        else:
            gap = abs(o_value - opt) / abs(opt)
        if n_objects is None:
            compression = None
        else:
            compression = 1 - (n_objects / n_events)
        line = (
            seed, name, price_signal, n_agents, o_value,
            gap, compression, time, config.name
        )
        return line

    # Run optimization
    opt, t_opt = run_opt(agents, costs, config)
    result.append( write_result_line("OPTIMAL", opt, t_opt, n_events) )

    # Uncontrolled
    o_unctr, t_unctr = run_uncontrolled(agents, costs, config)
    result.append( write_result_line("UNCONTROLLED", o_unctr, t_unctr, n_events) )

    # Represenative profiles
    o_rep, t_rep, n_rep = run_rep(agents, costs, config, seed, 500, "LL")
    result.append( write_result_line("REP", o_rep, t_rep, n_rep) )

    # FO
    o_fo, t_fo, n_fo = run_fo(agents, costs, config, 15, 7)
    result.append( write_result_line("FO", o_fo, t_fo, n_fo) )

    # DFO
    for num_sampels in [2, 4, 6, 8]:
        o_dfo, t_dfo, n_dfo = run_dfo(agents, costs, config, 3, 48, num_sampels)
        result.append( write_result_line(f"DFO_{num_sampels}", o_dfo, t_dfo, n_dfo) )

    # VB
    o_vb, t_vb, n_vb = run_vb(
        agents, costs, config, "LL", None, None, None, False, False, None, None
    )
    result.append( write_result_line("VB", o_vb, t_vb, n_vb) )

    o_vbd, t_vbd, n_vbd = run_vb(
        agents, costs, config, "ED", None, None, None, False, False, None, None
    )
    result.append( write_result_line("VB-ED", o_vbd, t_vbd, n_vbd) )

    o_vbf, t_vbf, n_vbf = run_vb(
        agents, costs, config, "LL", 0.8, 0.6, 0.15, True, False, None, None
    )
    result.append( write_result_line("VB-FSOE", o_vbf, t_vbf, n_vbf) )

    o_vbg, t_vbg, n_vbg = run_vb(
        agents, costs, config, "LL", None, None, None, False, True, 33, 8
    )
    result.append( write_result_line("VB-GRPD", o_vbg, t_vbg, n_vbg) )

    # Save results
    df = pd.DataFrame(result, columns=res_columns)
    df.to_csv(
        f"../results/data/Benchmark_{seed}_{n_agents}_{price_signal}_{config.name}.csv",
        sep=";"
    )

def run(trips, reps, fn_price, fn_gen, fn_dem, config):
    """
    Run benchmark.
    """
    # If you don't want to download data from www.smard.de use:
    # price_signals = [cts.PriceSignal.SINE]
    # You can then set: fn_price, fn_gen, fn_dem = None, None, None
    price_signals = [cts.PriceSignal.SINE, cts.PriceSignal.REAL, cts.PriceSignal.FUTURE]
    n_agent_options = [100, 1000, 10_000, 100_000]
    n_runs = reps * len(n_agent_options) * len(price_signals)
    runs_done = 0
    for n_agents in n_agent_options:
        for price_signal in price_signals:
            for seed in range(reps):
                run_iteration(trips, n_agents, seed, price_signal, fn_price, fn_gen, fn_dem, config)
                runs_done += 1
                utils.print_progress(runs_done, n_runs)

if __name__ == "__main__":
    REPS = 10

    mid_trips = pd.read_csv(c.MID_LOCATION, sep=";", decimal=",")
    
    # Run
    run(
        mid_trips, REPS, python_code.config.FN_PRICE,
        python_code.config.FN_GEN, python_code.config.FN_DEM, python_code.config.defaultConfig
    )
    