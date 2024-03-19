"""
Parameter tuning
"""
from datetime import datetime
from io import StringIO
import utils
import config as c
import cost_ts as cts
import raw_python_impls.fo as fo
import dfo
import raw_python_impls.virtual_battery as vb
import raw_python_impls.disaggregation as disagg
import opt_utils as outils
import raw_python_impls.fo_grouping as fogrp
import midcrypt
import pandas as pd

def tune_fo(reps, n_agents, trips, fn_price):
    """
    Parameter tuning for FOs. Tunes earliest start threshold and time-flexibility threshold.
    """
    result = []
    res_columns = ["Seed", "Earliest start time", "Time flexibility", "Gap", "Compression"]
    search_area_est = list(range(0, 96, 1))
    search_area_tf  = list(range(0, 96, 1))

    n_runs = reps * len(search_area_est) * len(search_area_tf)
    runs_done = 0

    for seed in range(reps):
        agents, costs = utils.prepare_run(
            trips, seed, n_agents, cts.PriceSignal.REAL, fn_price, None, None
        )
        opt = utils.get_opt(agents, costs)
        n_events = utils.get_event_count(agents)
        for est in search_area_est:
            for tf in search_area_tf:
                load, n_objects = fo.pipeline(agents, c.ETA, c.SOC_START, c.DELTA_T, costs, est, tf)
                o_value = utils.calc_total_costs(load, costs)
                result.append(
                    (seed, est, tf,  abs(o_value - opt) / abs(opt), 1 - (n_objects / n_events))
                )
                runs_done += 1
                utils.print_progress(runs_done, n_runs)
    df_fo = pd.DataFrame(result, columns=res_columns)
    df_fo.to_csv("results/Param_tuning_FO.csv", sep=";")

def tune_dfo(reps, n_agents, trips, fn_price):
    """
    Parameter tuning for DFOs. Tunes earliest start threshold and stop threshold.
    """
    result = []
    res_columns = ["Seed", "Earliest start time", "Stop time", "Num Sampels", "Gap", "Compression"]
    search_area_est = list(range(4)) + list(range(4, 32, 4)) + list(range(32, 97, 8))
    search_area_st  = list(range(4)) + list(range(4, 32, 4)) + list(range(32, 97, 8))
    search_area_num_samples = [4]

    n_runs = reps * len(search_area_est) * len(search_area_st) * len(search_area_num_samples)
    runs_done = 0

    for seed in range(reps):
        agents, costs = utils.prepare_run(
            trips, seed, n_agents, cts.PriceSignal.REAL, fn_price, None, None
        )
        opt = utils.get_opt(agents, costs)
        n_events = utils.get_event_count(agents)
        for est in search_area_est:
            for st in search_area_st:
                for num_samples in search_area_num_samples:
                    load, n_objects  = dfo.pipeline(
                        agents, c.ETA, c.DELTA_T, costs, c.SOC_START, 1,
                        est=est, lst=st, num_samples=num_samples
                    )
                    o_value = utils.calc_total_costs(load, costs)
                    result.append(
                        (seed, est, st, num_samples,
                         abs(o_value - opt) / abs(opt), 1 - (n_objects / n_events))
                    )
                    runs_done += 1
                    utils.print_progress(runs_done, n_runs)
    df_dfo = pd.DataFrame(result, columns=res_columns)
    df_dfo.to_csv("results/Param_tuning_DFO.csv", sep=";")

def tune_vbfsoe(reps, n_agents, trips, fn_price):
    """
    Parameter tuning for vb with p_max = f(soe).
    Tunes the PWL-Constraing that defines f(soe).
    """
    result = []
    res_columns = ["Seed", "PWL Mid-Point x", "PWL Mid-Point y", "y-Offset", "Gap"]
    search_area_mpx =  [i/100 for i in range(0, 100, 5)]
    search_area_mpy  = [i/100 for i in range(0, 100, 5)]
    search_area_offset  = [i/100 for i in range(0, 100, 5)]

    n_runs = reps * len(search_area_mpx) * len(search_area_mpy) * len(search_area_offset)
    runs_done = 0

    for seed in range(reps):
        agents, costs = utils.prepare_run(
            trips, seed, n_agents, cts.PriceSignal.REAL, fn_price, None, None
        )
        opt = utils.get_opt(agents, costs)
        for mpx in search_area_mpx:
            for mpy in search_area_mpy:
                if mpy < 1 - mpx:
                    runs_done += len(search_area_offset)
                    continue # Would lead to non-linear constraint
                for offset in search_area_offset:
                    if offset > mpy:
                        runs_done += 1
                        continue # Would lead to non-linear constraint
                    load = utils.to_ts(
                        vb.pipeline(
                            agents=agents, costs=costs, eta=c.ETA, soc_start=c.SOC_START,
                            delta_t=c.DELTA_T, disagg_option=disagg.Option.LAXITY,
                            support_point=(mpx, mpy), offset_at_full=offset, with_fsoe=True
                        ),
                        len(costs)
                    )
                    o_value = utils.calc_total_costs(load, costs)
                    result.append( (seed, mpx, mpy, offset, abs(o_value - opt) / abs(opt)) )
                    runs_done += 1
                    utils.print_progress(runs_done, n_runs)
    df_vb_fsoe = pd.DataFrame(result, columns=res_columns)
    df_vb_fsoe.to_csv("results/Param_tuning_VBFSOE.csv", sep=";")

def tune_vbgrpd(reps, n_agents, trips, fn_price):
    """
    Parameter tuning for VB-grpd. Tunes earliest start threshold and stop threshold.
    """
    result = []
    res_columns = ["Seed", "Earliest start time", "Stop time", "Gap", "Compression"]
    search_area_est = [i for i in range(0, 96, 1)]
    search_area_st  = [i for i in range(0, 96, 1)]

    n_runs = reps * len(search_area_est) * len(search_area_st)
    runs_done = 0

    for seed in range(reps):
        agents, costs = utils.prepare_run(
            trips, seed, n_agents, cts.PriceSignal.REAL, fn_price, None, None
        )
        opt = utils.get_opt(agents, costs)
        n_events = utils.get_event_count(agents)
        for est in search_area_est:
            for st in search_area_st:
                load, n_objects  = vb.pipeline_grpd(
                    agents=agents, costs=costs, eta=c.ETA, soc_start=c.SOC_START, delta_t=c.DELTA_T,
                    est=est, lst=st, disagg_option=disagg.Option.LAXITY, support_point=None,
                    offset_at_full=None, with_fsoe=False
                )
                o_value = utils.calc_total_costs(load, costs)
                result.append(
                    (seed, est, st,
                     abs(o_value - opt) / abs(opt), 1 - (n_objects / n_events))
                )
                runs_done += 1
                utils.print_progress(runs_done, n_runs)
    df_vb_grpd = pd.DataFrame(result, columns=res_columns)
    df_vb_grpd.to_csv("results/Param_tuning_VBFGRPD.csv", sep=";")

def eval_compression(reps, trips, thresh_start, thresh_stop):
    """
    Check the compression for different numbers of EV with the given start and stop tresholds.
    """
    result = []
    res_columns = ["Seed", "Number of Agents", "Number charging events", "Aggregated number", "Compression"]
    search_area_agents = [10**(i/4) for i in range(2*4, 7*4+1)]

    n_runs = reps * len(search_area_agents)
    runs_done = 0

    for seed in range(reps):
        for n_agents in search_area_agents:
            n_agents = int(n_agents)
            agents, _ = utils.prepare_run(
                trips, seed, n_agents, cts.PriceSignal.SINE, None, None, None
            )

            # Get charging events
            events = []
            for agent in agents:
                events.extend(
                    outils.get_chrg_events_restricted(
                        agent=agent, eta=c.ETA, soc_start=c.SOC_START, delta_t=c.DELTA_T
                    )
                )
            n_events = utils.get_event_count(agents)

            # Group
            groups = fogrp.pre_group(
                events, thresh_start, thresh_stop, lambda x: x.start, lambda x: x.stop
            )
            opt_groups = fogrp.optimize_groups(
                groups, thresh_start, thresh_stop, lambda x: x.start, lambda x: x.stop
            )

            result.append([seed, n_agents, n_events, len(opt_groups), 1 - len(opt_groups)/n_events])
            runs_done += 1
            utils.print_progress(runs_done, n_runs)
    df_comp = pd.DataFrame(result, columns=res_columns)
    df_comp.to_csv("results/Compression_Eval.csv", sep=";")

if __name__ == "__main__":
    REPS = 5
    N_AGENTS = 1000
    # v Change this v
    FN_PRICE = "data/Gro_handelspreise_202201010000_202301012359_Viertelstunde.csv"

    # Load MID data, change this path to your copy of "Wege.csv" of the MID 2017
    mid_trips = midcrypt.fetchFromDir("/home/ubuntu/J/MID/MiD2017_Lokal_Wege.csv.encrypted")
    mid_trips = pd.read_csv(StringIO(mid_trips), sep=";", decimal=",")

    # FO
    print(f"---- Tuning FO ----\t{datetime.now()}", flush=True)
    tune_fo(REPS, N_AGENTS, mid_trips, FN_PRICE)

    # DFO
    print(f"---- Tuning DFO ----\t{datetime.now()}", flush=True)
    tune_dfo(REPS, N_AGENTS, mid_trips, FN_PRICE)

    # VB - fsoe
    print(f"---- Tuning VB-FSOE ----\t{datetime.now()}", flush=True)
    tune_vbfsoe(REPS, N_AGENTS, mid_trips, FN_PRICE)

    # VB - Grpd
    print(f"---- Tuning VB-GRPD ----\t{datetime.now()}", flush=True)
    tune_vbgrpd(REPS, N_AGENTS, mid_trips, FN_PRICE)

    # Evaluate compression
    print(f"---- Evaluate Compression ----\t{datetime.now()}", flush=True)
    eval_compression(REPS, mid_trips, 3, 48)
