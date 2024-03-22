use std::cmp::Ordering;
use std::collections::BinaryHeap;
use crate::chrg_event::ChrgEvent;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[derive(PartialEq)]
pub struct ChrgEventWState {
    sd: ChrgEvent, // Static data
    energy_curr: f64,
    eta: f64,
    delta_t: f64
}
impl Eq for ChrgEventWState {}

pub trait PrioEvent: Ord {
    fn from_event(event: ChrgEventWState) -> Self;
    fn get_event(self) -> ChrgEventWState;
    fn get_event_as_ref(&self) -> &ChrgEventWState;
    fn get_event_as_mut(&mut self) -> &mut ChrgEventWState;
}

#[derive(PartialEq, Eq)]
pub struct EDWrapper {
    event: ChrgEventWState
}
impl PrioEvent for EDWrapper {
    fn from_event(event: ChrgEventWState) -> Self {
        Self { event }
    }

    fn get_event(self) -> ChrgEventWState {
        return self.event;
    }

    fn get_event_as_ref(&self) -> &ChrgEventWState {
        return &self.event;
    }

    fn get_event_as_mut(&mut self) -> &mut ChrgEventWState {
        return &mut self.event;
    }
}
impl Ord for EDWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        other.event.sd.stop.cmp(&self.event.sd.stop)
    }
}
impl PartialOrd for EDWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

#[derive(PartialEq, Eq)]
pub struct LLWrapper {
    event: ChrgEventWState
}
impl PrioEvent for LLWrapper {
    fn from_event(event:ChrgEventWState) -> Self {
        Self { event }
    }

    fn get_event(self) -> ChrgEventWState {
        return self.event;
    }
    
    fn get_event_as_ref(&self) -> &ChrgEventWState {
        return &self.event;
    }

    fn get_event_as_mut(&mut self) -> &mut ChrgEventWState {
        return &mut self.event;
    }
}
impl Ord for LLWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        // t can be removed because it doesn't change the ordering
        let laxity_a = (self.event.sd.stop as f64) - t_until_charged(&self.event);
        let laxity_b = (other.event.sd.stop as f64) - t_until_charged(&other.event);
        laxity_b.partial_cmp(&laxity_a).unwrap()
    }
}
impl PartialOrd for LLWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

fn t_until_charged(event: &ChrgEventWState) -> f64 {
    (event.sd.energy_departure - event.energy_curr) / (event.sd.p_max * event.eta * event.delta_t) 
}

fn must_charge(event: &ChrgEventWState, t: i32) -> bool {
    let t_flex = (event.sd.stop - t) as f64;
    let t_chrg = t_until_charged(event);
    return t_flex - t_chrg < 1.0;
}

fn do_must_charge(event: &ChrgEventWState, t: usize, power_signal: &mut Vec<f64>, load: &mut Vec<f64>) {
    let mut energy_curr = event.energy_curr;
    for t_me in t..(event.sd.stop as usize) {
        // Power possible
        let p_tofill = (event.sd.energy_departure - energy_curr) / (event.eta * event.delta_t);
        let p = f64::min(event.sd.p_max, p_tofill);
        
        // Power realized
        let e = p * event.eta * event.delta_t;
        power_signal[t_me] -= p;
        load[t_me] += p;
        energy_curr += e;

        if energy_curr >= event.sd.energy_departure {
            break;
        }
    }
}

pub fn disaggregate_prio_based<T>(
    events: Vec<ChrgEvent>, power_signal: Vec<f64>, eta: f64, delta_t: f64
) -> Vec<f64>
where T: PrioEvent {
    let n_t = power_signal.len();
    let mut load = vec![0.0; n_t];
    let mut power_signal = power_signal;

    // Group events by arrival
    let mut event_by_arrival = Vec::with_capacity(n_t);
    for _ in 0..n_t {
        event_by_arrival.push(Vec::new())
    }
    for event in events {
        if event.p_max <= 0.0 { continue; } // Filter events without charging option
        let start = event.start as usize;
        let energy_curr = event.energy_arrival;
        let prio_event = T::from_event(ChrgEventWState{sd: event, energy_curr, delta_t, eta});
        event_by_arrival[start].push(prio_event);
    }

    // Disaggregate
    let mut active_events: BinaryHeap<T> = BinaryHeap::new();
    for (t, arrivals) in event_by_arrival.into_iter().enumerate() {
        // Check must charge 
        // Old Events
        for prio_obj in active_events.iter() {
            let event = prio_obj.get_event_as_ref();
            if must_charge(event, t as i32) {
                do_must_charge(event, t, &mut power_signal, &mut load)
            }
        }
        active_events.retain( |prio_obj: &T|
            !must_charge(prio_obj.get_event_as_ref(), t as i32)
        );
        // New Events
        for prio_obj in arrivals {
            let event = prio_obj.get_event_as_ref();
            if must_charge(event, t as i32) {
                do_must_charge(event, t, &mut power_signal, &mut load)
            } else {
                active_events.push(prio_obj);
            }
        }

        let mut repush = Vec::new();
        while active_events.len() > 0 {
            let mut prio_obj = active_events.pop().unwrap();
            let event = prio_obj.get_event_as_mut();

            let t_flex = (event.sd.stop - (t as i32)) as f64;
            let laxity = t_flex - t_until_charged(&event);

            // Determine power of event at time t
            // Case 1: No power needed by event or signal
            if (laxity >= 1.0) && (power_signal[t] <= 0.0) {
                repush.push(prio_obj);
                break;
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

            if ((t as i32) < event.sd.stop) && (event.sd.energy_departure > event.energy_curr) {
                repush.push(prio_obj);
            }
        }
        
        // Reschedule unfinished events
        for prio_obj in repush {
            active_events.push(prio_obj)
        }
    }
    return load;
}

#[pyfunction]
pub fn disaggregate (events: Vec<ChrgEvent>, power_signal: Vec<f64>, eta: f64, delta_t: f64, method: &str) -> PyResult<Vec<f64>> {
    return match method {
        "LL" => Ok(disaggregate_prio_based::<LLWrapper>(events, power_signal, eta, delta_t)),
        "ED" => Ok(disaggregate_prio_based::<EDWrapper>(events, power_signal, eta, delta_t)),
        _ => Err(PyValueError::new_err("Method not supported!"))
    }
}