use crate::chrg_event::ChrgEvent;
use pyo3::prelude::*;

#[pyfunction]
pub fn run_unctr(events: Vec<ChrgEvent>, eta: f64, delta_t: f64, n_t: usize) -> Vec<f64> {
    let mut rslt = vec![0.0; n_t];
    for event in events {
        let mut soe = event.energy_arrival;
        for t in event.start..event.stop {
            let e_charge = f64::max(0.0, f64::min(event.p_max * eta * delta_t, event.energy_departure - soe));
            soe += e_charge;

            if e_charge <= 0.0 { break; }
            rslt[t as usize] += e_charge / (delta_t * eta);
        }
    }
    return rslt;
}