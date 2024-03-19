use std::cmp::Ordering;
use crate::chrg_event::ChrgEvent;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

pub struct ChrgEventWState<'a> {
    sd: &'a ChrgEvent, // Static data
    energy_curr: f64
}

fn t_until_charged(event: &ChrgEventWState, eta: f64, delta_t: f64) -> f64 {
    (event.sd.energy_departure - event.energy_curr) / (event.sd.p_max * eta * delta_t) 
}

fn must_charge(event: &ChrgEventWState, t: i32, eta: f64, delta_t: f64) -> bool {
    let t_flex = (event.sd.stop - t) as f64;
    let t_chrg = t_until_charged(event, eta, delta_t);
    return t_flex - t_chrg < 1.0;
}

pub fn prio_earliest_departure(
    event_a: &ChrgEventWState, event_b: &ChrgEventWState, t: i32, eta: f64, delta_t: f64
) -> Ordering {
    let a_must = must_charge(event_a, t, eta, delta_t);
    let b_must = must_charge(event_b, t, eta, delta_t);
    if a_must && b_must {
        Ordering::Equal
    } else if a_must {
        Ordering::Less
    } else if b_must {
        Ordering::Greater
    } else {
        event_a.sd.stop.cmp(&event_b.sd.stop)
    }
}

pub fn prio_least_laxity(
    event_a: &ChrgEventWState, event_b: &ChrgEventWState, t: i32, eta: f64, delta_t: f64
) -> Ordering {
    let a_must = must_charge(event_a, t, eta, delta_t);
    let b_must = must_charge(event_b, t, eta, delta_t);
    if a_must && b_must {
        Ordering::Equal
    } else if a_must {
        Ordering::Less
    } else if b_must {
        Ordering::Greater
    } else {
        let t_flex_a = (event_a.sd.stop - t) as f64;
        let t_flex_b = (event_b.sd.stop - t) as f64;
        let laxity_a = t_flex_a - t_until_charged(event_a, eta, delta_t);
        let laxity_b = t_flex_b - t_until_charged(event_b, eta, delta_t);
        laxity_a.partial_cmp(&laxity_b).unwrap()
    }
}

pub fn disaggregate_prio_based(
    events: Vec<ChrgEvent>, power_signal: Vec<f64>, eta: f64, delta_t: f64,
    priority_metric: fn(&ChrgEventWState, &ChrgEventWState, i32, f64, f64) -> Ordering
) -> Vec<f64> {
    let n_t = power_signal.len();
    let mut load = vec![0.0; n_t];
    let mut power_signal = power_signal;

    // Group events by arrival
    let mut event_by_arrival = Vec::with_capacity(n_t);
    for _ in 0..n_t {
        event_by_arrival.push(Vec::new())
    }
    for event in &events {
        let event_w_state = ChrgEventWState{sd: event, energy_curr: event.energy_arrival};
        event_by_arrival[event.start as usize].push(event_w_state);
    }

    // Disaggregate
    let mut active_events = Vec::new();
    for (t, arrivals) in event_by_arrival.into_iter().enumerate() {
        active_events.extend(arrivals);

        // Sort - Reversed: Highest priority is the last item
        let sorter = |event_a: &ChrgEventWState, event_b: &ChrgEventWState| -> Ordering {
            priority_metric(event_b, event_a, t as i32, eta, delta_t)
        };
        active_events.sort_by(sorter);

        for event in &mut active_events {
            let t_flex = (event.sd.stop - (t as i32)) as f64;
            let laxity = t_flex - t_until_charged(&event, eta, delta_t);

            // Determine power of event at time t
            // Case 1: No power needed by event or signal
            if (laxity >= 1.0) && (power_signal[t] <= 0.0) {
                continue;
            }
            // Case 2: Power needed
            // Needed by event:
            let p_event = if laxity <= 0.0 {
                event.sd.p_max
            } else if laxity < 1.0 {
                (1.0 - laxity) * event.sd.p_max
            } else {
                0.0
            };
            // Needed total
            let p_needed = f64::max(p_event, power_signal[t]);
            // Power possible
            let p_tofill = (event.sd.energy_departure - event.energy_curr) / (eta * delta_t);
            let p_possible = f64::min(event.sd.p_max, p_tofill);
            // Power realized
            let p = f64::min(p_needed, p_possible);
            let e = p * eta * delta_t;
            power_signal[t] -= p;
            load[t] += p;
            event.energy_curr += e;
        }

        // Throw away finished events
        active_events.retain(|event| {
            ((t as i32) < event.sd.stop) && (event.sd.energy_departure > event.energy_curr)
        })
    }
    return Vec::new();
}

#[pyfunction]
pub fn disaggregate (events: Vec<ChrgEvent>, power_signal: Vec<f64>, eta: f64, delta_t: f64, method: &str) -> PyResult<Vec<f64>> {
    let prio_metric = match method {
        "LL" => prio_least_laxity,
        "ED" => prio_earliest_departure,
        _ => return Err(PyValueError::new_err("Method not supported!"))
    };

    return Ok(disaggregate_prio_based(events, power_signal, eta, delta_t, prio_metric));
}