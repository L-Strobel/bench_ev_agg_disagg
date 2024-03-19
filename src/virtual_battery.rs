use crate::chrg_event::ChrgEvent;
use crate::fo_grouping as fogrp;
use pyo3::prelude::*;

#[pyclass]
#[derive(FromPyObject)]
pub struct VBSlice {
    #[pyo3(get)]
    capability: f64,
    #[pyo3(get)]
    e_arrival: f64,
    #[pyo3(get)]
    e_departure: f64,
    #[pyo3(get)]
    p_avg: f64,
    #[pyo3(get)]
    p_min: f64,
    #[pyo3(get)]
    e_max: f64,
    #[pyo3(get)]
    e_min: f64
}

#[pyfunction]
pub fn get_vb_params(events: Vec<ChrgEvent>, n_t: usize, eta: f64, delta_t: f64) -> Vec<VBSlice> {
    let mut slices = Vec::with_capacity(n_t);
    for _ in 0..n_t {
        slices.push( VBSlice {capability: 0.0, e_arrival: 0.0, e_departure: 0.0, p_avg: 0.0, p_min: 0.0, e_max: 0.0, e_min: 0.0} );
    }
    
    for event in events {
        slices[event.start as usize].e_arrival += event.energy_arrival;
        slices[event.stop as usize].e_departure += event.energy_departure;

        let e_event_min = event.calc_e_min(eta, delta_t);
        let e_event_max = event.calc_e_max(eta, delta_t);

        let e_event_needed = event.energy_departure - event.energy_arrival;
        let e_event_possible = (event.stop - event.start) as f64 * delta_t * eta * event.p_max;

        // Constant power that fullfills demand
        let p_event_avg = e_event_needed / ((event.stop - event.start) as f64 * delta_t * eta);

        for t in event.start..event.stop {
            let t = t as usize;
            let slice = &mut slices[t];

            // Energy limits
            slice.e_min += e_event_min[t];
            slice.e_max += e_event_max[t];

            // Power limits
            slice.capability += event.p_max;
            
            // Store p_avg. Needed to gurantee flexibility of piece-wise-constraint
            slice.p_avg += p_event_avg;
        }

        // Event needs constant power?
        if e_event_needed >= e_event_possible {
            for t in event.start..event.stop {
                let t = t as usize;
                let slice = &mut slices[t];
                slice.p_min += event.p_max;
            }
        }
    }
    return slices;
}

#[pyfunction]
pub fn group (events: Vec<ChrgEvent>, est: i32, lst: i32) -> Vec<Vec<ChrgEvent>> {
    fn get_x(o: &ChrgEvent) -> i32 { o.start }
    fn get_y(o: &ChrgEvent) -> i32 { o.stop }
    let groups = fogrp::pre_group(events, est, lst, get_x, get_y);
    let optimized_groups = fogrp::optimize_groups(groups, est, lst, get_x, get_y);
    let grouped = optimized_groups.into_iter().map(|x| x.cells.into_iter().flat_map(|c| c.objects).collect()).collect();
    return grouped;
}