"""
Run benchmark
"""
import timeit
from io import StringIO
import midcrypt
import pandas as pd
import utils
import config as c
import cost_ts as cts
import raw_python_impls.fo as fo
import dfo
import raw_python_impls.uncontrolled as uncontrolled
import rep_profile as rep
import raw_python_impls.virtual_battery as vb
import raw_python_impls.disaggregation as disagg

def run_opt(agents, costs):
    """
    Run optimal
    """
    t_start = timeit.default_timer()
    opt = utils.get_opt(agents, costs)
    t_stop = timeit.default_timer()
    return opt, t_stop - t_start

def run_uncontrolled(agents, costs):
    """
    Run uncontrolled
    """
    t_start = timeit.default_timer()
    load = utils.to_ts(
        uncontrolled.run(agents=agents, eta=c.ETA, soc_start=c.SOC_START, delta_t=c.DELTA_T),
        len(costs)
    )
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start

def run_rep(agents, costs, seed, n_profiles, disagg_option):
    """
    Run represenative profile
    """
    t_start = timeit.default_timer()
    load = utils.to_ts(
        rep.pipeline(
            agents, seed, n_profiles, disagg_option, c.ETA,
            c.SOC_START, c.DELTA_T, costs
        ),
        len(costs)
    )
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start, n_profiles

def run_fo(agents, costs, est, tf):
    """
    Run FO
    """
    t_start = timeit.default_timer()
    load, n_objects = fo.pipeline(agents, c.ETA, c.SOC_START, c.DELTA_T, costs, est, tf)
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start, n_objects

def run_dfo(agents, costs, est, st, num_samples):
    """
    Run DFO
    """
    t_start = timeit.default_timer()
    try:
        load, n_objects  = dfo.pipeline(
            agents, c.ETA, c.DELTA_T, costs, c.SOC_START, 1,
            est=est, lst=st, num_samples=num_samples
        )
        t_stop = timeit.default_timer()
        o_value = utils.calc_total_costs(load, costs)
    except AssertionError:
        t_stop = timeit.default_timer()
        load = None
        n_objects = None
        o_value = None
    return o_value, t_stop - t_start, n_objects

def run_vb(agents, costs, disagg_option, mpx, mpy, offset, with_fsoe, grpd, est, st):
    """
    Run virtual battery
    """
    t_start = timeit.default_timer()
    if grpd:
        load, n_objects  = vb.pipeline_grpd(
            agents=agents, costs=costs, eta=c.ETA, soc_start=c.SOC_START, delta_t=c.DELTA_T,
            est=est, lst=st, disagg_option=disagg_option, support_point=(mpx, mpy),
            offset_at_full=offset, with_fsoe=with_fsoe
        )
    else:
        load = utils.to_ts(
            vb.pipeline(
                agents=agents, costs=costs, eta=c.ETA, soc_start=c.SOC_START,
                delta_t=c.DELTA_T, disagg_option=disagg_option,
                support_point=(mpx, mpy), offset_at_full=offset, with_fsoe=with_fsoe
            ),
            len(costs)
        )
        n_objects = 1
    t_stop = timeit.default_timer()
    o_value = utils.calc_total_costs(load, costs)
    return o_value, t_stop - t_start, n_objects

def run_iteration(trips, n_agents, seed, price_signal, fn_price, fn_gen, fn_dem):
    """
    Run benchmark for one seed, agent number, and price signal combination.
    """
    result = []
    res_columns = [
        "Seed", "Method", "Price Signal", "N_Agents",
        "Objective Value", "Gap", "Compression", "Runtime_s"
    ]

    # Prepare
    agents, costs = utils.prepare_run(trips, seed, n_agents, price_signal, fn_price, fn_gen, fn_dem)
    n_events = utils.get_event_count(agents)

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
            gap, compression, time
        )
        return line

    # Run optimization
    opt, t_opt = run_opt(agents, costs)
    result.append( write_result_line("OPTIMAL", opt, t_opt, n_events) )

    # Uncontrolled
    o_unctr, t_unctr = run_uncontrolled(agents, costs)
    result.append( write_result_line("UNCONTROLLED", o_unctr, t_unctr, n_events) )

    # Represenative profiles
    o_rep, t_rep, n_rep = run_rep(agents, costs, seed, 500, disagg.Option.LAXITY)
    result.append( write_result_line("REP", o_rep, t_rep, n_rep) )

    # FO
    o_fo, t_fo, n_fo = run_fo(agents, costs, 15, 7)
    result.append( write_result_line("FO", o_fo, t_fo, n_fo) )

    # DFO
    for num_sampels in [2, 4, 6, 8]:
        o_dfo, t_dfo, n_dfo = run_dfo(agents, costs, 3, 48, num_sampels)
        result.append( write_result_line(f"DFO_{num_sampels}", o_dfo, t_dfo, n_dfo) )

    # VB
    o_vb, t_vb, n_vb = run_vb(
        agents, costs, disagg.Option.LAXITY, None, None, None, False, False, None, None
    )
    result.append( write_result_line("VB", o_vb, t_vb, n_vb) )

    o_vbd, t_vbd, n_vbd = run_vb(
        agents, costs, disagg.Option.DEPARTURE, None, None, None, False, False, None, None
    )
    result.append( write_result_line("VB-ED", o_vbd, t_vbd, n_vbd) )

    o_vbf, t_vbf, n_vbf = run_vb(
        agents, costs, disagg.Option.LAXITY, 0.8, 0.6, 0.15, True, False, None, None
    )
    result.append( write_result_line("VB-FSOE", o_vbf, t_vbf, n_vbf) )

    o_vbg, t_vbg, n_vbg = run_vb(
        agents, costs, disagg.Option.LAXITY, None, None, None, False, True, 33, 8
    )
    result.append( write_result_line("VB-GRPD", o_vbg, t_vbg, n_vbg) )

    # Save results
    df = pd.DataFrame(result, columns=res_columns)
    df.to_csv(f"results/Benchmark_{seed}_{n_agents}_{price_signal}.csv", sep=";")

def run(trips, reps, fn_price, fn_gen, fn_dem):
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
                run_iteration(trips, n_agents, seed, price_signal, fn_price, fn_gen, fn_dem)
                runs_done += 1
                utils.print_progress(runs_done, n_runs)

if __name__ == "__main__":
    REPS = 10

    # v Change these v
    FN_PRICE = "data/Gro_handelspreise_202201010000_202301012359_Viertelstunde.csv"
    FN_GEN = "data/Realisierte_Erzeugung_202201010000_202301012359_Viertelstunde.csv"
    FN_DEM = "data/Realisierter_Stromverbrauch_202201010000_202301012359_Viertelstunde.csv"

    # Load MID data, change this path to your copy of "Wege.csv" of the MID 2017
    mid_trips = midcrypt.fetchFromDir("/home/leo/J/MID/MiD2017_Lokal_Wege.csv.encrypted")
    mid_trips = pd.read_csv(StringIO(mid_trips), sep=";", decimal=",")
    run(mid_trips, REPS, FN_PRICE, FN_GEN, FN_DEM)
    