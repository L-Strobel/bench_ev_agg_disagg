use pyo3::prelude::*;

#[pyclass]
#[derive(FromPyObject)]
pub struct ChrgEvent {
    #[pyo3(get)]
    pub start: i32,
    #[pyo3(get)]
    pub stop: i32,
    #[pyo3(get)]
    pub p_max: f64,
    #[pyo3(get)]
    pub energy_arrival: f64,
    #[pyo3(get)]
    pub energy_departure: f64,
    #[pyo3(get)]
    pub capacity: f64
}

#[pymethods]
impl ChrgEvent {
    #[new]
    pub fn new(start: i32, stop: i32, p_max: f64, energy_arrival: f64, energy_departure: f64, capacity: f64) -> Self {
        ChrgEvent { start, stop, p_max, energy_arrival, energy_departure, capacity }
    }

    pub fn calc_e_max(&self, eta: f64, delta_t: f64) -> Vec<f64> {
        // Charge as much as possible
        let len_event = (self.stop - self.start) as usize;
        let mut e_max = Vec::with_capacity(len_event);
        let mut soe = self.energy_arrival;
        e_max.push(soe);
        for _ in 0..len_event {
            let e_charge = f64::max(0.0, f64::min(self.p_max * eta * delta_t, self.capacity - soe));
            soe += e_charge;
            e_max.push(soe); 
        }
        return e_max;
    }

    pub fn calc_e_min(&self, eta: f64, delta_t: f64) -> Vec<f64> {
        // Charge as little as possible
        let len_event = (self.stop - self.start) as usize;
        let mut e_min = Vec::with_capacity(len_event);
        let mut soe_rev = self.energy_departure;
        for _ in 0..len_event {
            e_min.push(soe_rev);
            let e_charge = f64::max(0.0, f64::min(self.p_max * eta * delta_t, soe_rev - self.energy_arrival));
            soe_rev -= f64::min(soe_rev, e_charge); // Don't let battery fall below zero
        }
        e_min.push(soe_rev);
        return e_min.into_iter().rev().collect();
    }
}
