use crate::chrg_event::ChrgEvent;
use crate::fo_grouping as fogrp;
use pyo3::prelude::*;
use itertools::izip;

#[pyclass]
#[derive(Copy, Clone, Debug)]
struct DFOVertex {
    #[pyo3(get)]
    d: f64,
    #[pyo3(get)]
    e: f64
}

impl DFOVertex {
    fn panik_cmp(lhs: &DFOVertex, rhs: &DFOVertex) -> std::cmp::Ordering {
        let l1_self = lhs.e + lhs.d;
        let l1_other = rhs.e + rhs.d;
        l1_self.partial_cmp(&l1_other).unwrap()
    }
}

/*pub struct ChrgFlexModel {
    pub start: i32,
    pub stop: i32,
    pub p_max: f64,
    pub energy_needed: f64
}*/

#[pyclass]
#[derive(Clone, Debug)]
struct Slice {
    #[pyo3(get)]
    vertices: Vec<DFOVertex>
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct DFO {
    #[pyo3(get)]
    start: i32,
    #[pyo3(get)]
    stop: i32,
    #[pyo3(get)]
    slices: Vec<Slice>
}

impl DFO {
    fn reduce(&mut self, eps: f64) {
        for slice in &mut self.slices {
            if slice.vertices.len() <= 3 {
                continue;
            }
            
            let mut endpoint = slice.vertices[0];
            let mut canditate = slice.vertices[1];
            let mut new_vertices = vec!(endpoint);
            for test_point in slice.vertices.iter().skip(2).chain(slice.vertices.iter().take(1)) {
                // Line check
                let a = endpoint.d - canditate.d;
                let b = test_point.e - canditate.e;
                let c = endpoint.e - canditate.e;
                let d = test_point.d - canditate.d;
                let v = a * b - c * d;
                // Points on a line?
                if v.abs() >= eps {
                    endpoint = canditate;
                    new_vertices.push(endpoint);
                // Does the line turn by 180° at the canditate?
                } else if (a * d >= 0.0) & (b* c >= 0.0) {
                    endpoint = canditate;
                    new_vertices.push(endpoint);
                }
                canditate = *test_point;
            }

            slice.vertices = new_vertices;
        } 
    }

    pub fn inner(cmodel: &ChrgEvent, num_samples: i32, eta: f64, delta_t: f64, eps: f64) -> DFO {
        let e_needed = cmodel.energy_departure - cmodel.energy_arrival;
        let n_t = cmodel.stop - cmodel.start;
        let mut slices = Vec::with_capacity(n_t as usize);

        let mut d_min = 0.0;
        let mut d_max = 0.0;
        for t in 0..n_t {
            let t_left = (n_t-t-1) as f64;

            let mut upper_vertices = Vec::new();
            let mut lower_vertices = Vec::new();
            if t == 0 {
                let v_max = f64::min(cmodel.p_max * eta * delta_t, e_needed);
                let v_min = f64::min(v_max, f64::max(0.0, e_needed - cmodel.p_max * eta * delta_t * t_left));
                upper_vertices.push( DFOVertex {d: 0.0, e: v_max } );
                lower_vertices.push(  DFOVertex {d: 0.0, e: v_min } );
            } else {
                for s in 0..num_samples {
                    let d = d_min + (s as f64) * (d_max - d_min) / ((num_samples - 1) as f64);
                    let energy_left = e_needed - d;
                    let v_max = f64::min(cmodel.p_max * eta * delta_t, energy_left);
                    let v_min = f64::min(v_max,  f64::max(0.0, energy_left - cmodel.p_max * eta * delta_t * t_left));
                    upper_vertices.push( DFOVertex {d: d, e: v_max } );
                    lower_vertices.push(  DFOVertex {d: d, e: v_min } );
                }
            } 
            // Update d_min and d_max
            let min_sum_vertex = lower_vertices.iter().copied().min_by( DFOVertex::panik_cmp).unwrap();
            d_min = min_sum_vertex.d + min_sum_vertex.e;
            let max_sum_vertex = upper_vertices.iter().copied().max_by( DFOVertex::panik_cmp).unwrap();
            d_max = max_sum_vertex.d + max_sum_vertex.e;
            // Store slice
            let vertices: Vec<DFOVertex> = upper_vertices.into_iter().chain(lower_vertices.into_iter().rev()).collect();
            slices.push(Slice {vertices});
        }
        let mut dfo = DFO {start: cmodel.start, stop: cmodel.stop, slices };
        dfo.reduce(eps);
        return dfo;
    }

    fn get_dep_energy_min_max(slice: &Slice) -> (f64, f64) {
        let val_fn = |x:&DFOVertex| x.d;
        let dmin = slice.vertices.iter().map(val_fn).fold(f64::INFINITY, f64::min); // Start folding with inf
        let dmax = slice.vertices.iter().map(val_fn).fold(0.0, f64::max);
        (dmin, dmax)
    }

    fn get_total_en_min_max(slice: &Slice) -> (f64, f64) {
        let val_fn = |x:&DFOVertex| x.d + x.e;
        let emin = slice.vertices.iter().map(val_fn).fold(f64::INFINITY, f64::min); // Start folding with inf
        let emax = slice.vertices.iter().map(val_fn).fold(0.0, f64::max);
        (emin, emax)
    }

    fn get_energy_min_max(slice: &Slice, d: f64, eps:f64) -> (f64, f64) {
        let mut emin = f64::INFINITY;
        let mut emax = 0.0;
        let mut last_v = slice.vertices.first().unwrap();

        for v in slice.vertices.iter().skip(1).chain(slice.vertices.iter().take(1)) {
            // d exactly on vertice v?
            if (d + eps >= v.d) && (v.d >= d - eps) {
                emin = f64::min(v.e, emin);            
                emax = f64::max(v.e, emax);
            // d between v and last_v?
            } else {
                let (left, right) = if v.d >= last_v.d {
                    (last_v, v)
                } else {
                    (v, last_v)
                };
                if (left.d <= d) && ( d <= right.d) {
                    // Interpolate position
                    let e = left.e + (d - left.d) * (right.e - left.e) / (right.d - left.d);
                    emin = f64::min(e, emin);
                    emax = f64::max(e, emax);
                }
            }
            last_v = v;
        }
        (emin, emax)
    }

    fn add_slices_infront(self: &mut DFO, n: usize) {
        let empty_slice = Slice{vertices: vec!{DFOVertex{e: 0.0, d: 0.0}} };
        let mut new_slices = vec![empty_slice; n];
        new_slices.append(&mut self.slices);
        self.slices = new_slices;
        self.start -= n as i32;
    }

    fn add_slices_end(self: &mut DFO, n: usize) {
        let (smin, smax) = DFO::get_total_en_min_max(self.slices.last().unwrap());
        let end_slice = Slice{vertices: vec!{DFOVertex{e: 0.0, d: smin}, DFOVertex{e: 0.0, d: smax}} };
        let mut new_slices = vec![end_slice; n];
        self.slices.append(&mut new_slices);
        self.stop += n as i32;
    }

    fn prepare_for_agg(this: &mut DFO, other: &mut DFO) {
        // Add slices in front if needed
        if this.start > other.start {
            this.add_slices_infront((this.start - other.start) as usize)
        } else if other.start > this.start {
            other.add_slices_infront((other.start - this.start) as usize)
        }
        // Add slices in back if needed
        if this.stop < other.stop {
            this.add_slices_end((other.stop - this.stop) as usize)
        } else if other.stop < this.stop {
            other.add_slices_end((this.stop - other.stop) as usize)
        }
    }

    pub fn agg2to1(dfo1: DFO, dfo2: DFO, num_samples: i32, eps: f64) -> DFO {
        let mut dfo1 = dfo1;
        let mut dfo2 = dfo2;
        DFO::prepare_for_agg(&mut dfo1, &mut dfo2);

        let n_t = dfo1.slices.len();

        let mut da_min = 0.0;
        let mut da_max = 0.0;
        let mut eta1 = 1.0;
        let mut eta2 = 1.0;
        let mut slices: Vec<Slice> = Vec::with_capacity(n_t as usize);

        for t in 0..n_t {
            let mut upper_vertices = Vec::new();
            let mut lower_vertices = Vec::new();

            let dfo1_slice = &dfo1.slices[t];
            let dfo2_slice = &dfo2.slices[t];

            let (d1_min, d1_max) = DFO::get_dep_energy_min_max(dfo1_slice);
            let (d2_min, d2_max) = DFO::get_dep_energy_min_max(dfo2_slice);
            let (s1_min, s1_max) = DFO::get_total_en_min_max(dfo1_slice);
            let (s2_min, s2_max) = DFO::get_total_en_min_max(dfo2_slice);

            for k in 0..num_samples {
                let k = k as f64; 
                let da = da_min + k * (da_max - da_min) / ((num_samples - 1) as f64);
                let d1 = d1_min + k * (d1_max - d1_min) / ((num_samples - 1) as f64);
                let d2 = d2_min + k * (d2_max - d2_min) / ((num_samples - 1) as f64);
                let (e1_min, e1_max) = DFO::get_energy_min_max(dfo1_slice, d1, eps);
                let (e2_min, e2_max) = DFO::get_energy_min_max(dfo2_slice, d2, eps);

                let (ea_min, ea_max) = if t < n_t - 1 {
                    // Handle dummy slices
                    let (s1_upper, s2_upper, s1_lower, s2_lower) = if ((s2_max - s2_min) <= 1e-9) || ((s1_max - s1_min)  <= 1e-9) {
                        let s1_upper = d1 + e1_max;
                        let s2_upper = d2 + e2_max;
                        let s1_lower = d1 + e1_min;
                        let s2_lower = d2 + e2_min;
                        (s1_upper, s2_upper, s1_lower, s2_lower)
                    // Normal case
                    } else {
                        let s1_range = s1_max - s1_min;
                        let s2_range = s2_max - s2_min;
    
                        // Upper bound s
                        let s1_upper_option1 = d1 + e1_max;
                        let s1_upper_option2 = (d2 + e2_max - s2_min) / s2_range * s1_range + s1_min;
                        let s1_upper = f64::min(s1_upper_option1, s1_upper_option2);
                        let s2_upper = (s1_upper - s1_min) / s1_range * s2_range + s2_min;
    
                        // Lower bound s
                        let s1_lower_option1 = d1 + e1_min;
                        let s1_lower_option2 = (d2 + e2_min - s2_min) / s2_range * s1_range + s1_min;
                        let s1_lower = f64::max(s1_lower_option1, s1_lower_option2);
                        let s2_lower = (s1_lower - s1_min) / s1_range * s2_range + s2_min;

                        (s1_upper, s2_upper, s1_lower, s2_lower)
                    };

                    let ea_max = s1_upper + s2_upper - da;
                    let ea_min = s1_lower + s2_lower - da;

                    if (e1_max - e1_min) != 0.0 {
                        eta1 = eta1  * (s1_upper - s1_lower) / (e1_max - e1_min);
                    }
                    if (e2_max - e2_min) != 0.0 {
                        eta2 = eta2  * (s2_upper - s2_lower) / (e2_max - e2_min);
                    }
                    (ea_min, ea_max)
                } else {
                    let ea_min = e1_min + e2_min;
                    let ea_max = e1_max + e2_max;
                    (ea_min, ea_max)
                };

                upper_vertices.push( DFOVertex{d: da, e: ea_max});
                lower_vertices.push( DFOVertex{d: da, e: ea_min});
            }
            // Store slice
            let vertices: Vec<DFOVertex> = upper_vertices.into_iter().chain(lower_vertices.into_iter().rev()).collect();
            let slice = Slice {vertices};
            (da_min, da_max) = DFO::get_total_en_min_max(&slice);
            slices.push(slice);
        }
        let mut dfo_a = DFO {start: dfo1.start, stop: dfo2.stop, slices};
        dfo_a.reduce(eps);
        return dfo_a
    }

    pub fn aggregate_n_m(dfos: Vec<DFO>, est: i32, lst: i32 , num_samples: i32, eps: f64) -> (Vec<DFO>, Vec<fogrp::Group<DFO>>) {
        fn get_x(o: &DFO) -> i32 { o.start }
        fn get_y(o: &DFO) -> i32 { o.stop }
        let groups = fogrp::pre_group(dfos, est, lst, get_x, get_y);
        let optimized_groups = fogrp::optimize_groups(groups, est, lst, get_x, get_y);
        let mut agg_objects = Vec::with_capacity(optimized_groups.len());
        for group in &optimized_groups {
            let mut dfo_iter = group.cells.iter().flat_map(|x| &x.objects);
            let mut dfo_a = dfo_iter.next().unwrap().clone();
            for dfo in dfo_iter {
                dfo_a = DFO::agg2to1(dfo.clone(), dfo_a, num_samples, eps);
            }
            dfo_a.reduce(eps);
            agg_objects.push(dfo_a)
        }
        return (agg_objects, optimized_groups)
    }

    fn disagg1to2(dfo1: &DFO, dfo2: &DFO, dfo_a: &DFO, y: &Vec<f64>, eps: f64) -> (Vec<f64>, Vec<f64>) {
        let mut sa = 0.0;
        let mut s1 = 0.0;
        let mut s2 = 0.0;

        let mut y1 = Vec::with_capacity(y.len());
        let mut y2 = Vec::with_capacity(y.len());

        let n_t = dfo_a.slices.len();

        for t in 0..n_t {
            if t < n_t -1 {
                sa += y[t];
                let (sa_min, sa_max) = DFO::get_total_en_min_max(&dfo_a.slices[t]);
                let mut f = 0.0;

                if sa_max -  sa_min != 0.0 {
                    f = (sa - sa_min) / (sa_max - sa_min);
                }

                let (s1_min, s1_max) = DFO::get_total_en_min_max(&dfo1.slices[t]);
                let y1_val = s1_min + f * (s1_max - s1_min) - s1;
                y1.push(y1_val);
                s1 += y1_val;

                let (s2_min, s2_max) = DFO::get_total_en_min_max(&dfo2.slices[t]);
                let y2_val = s2_min + f * (s2_max - s2_min) - s2;
                y2.push(y2_val);
                s2 += y2_val;
            } else {
                let (ea_min, ea_max) = DFO::get_energy_min_max(&dfo_a.slices[t], sa, eps);
                let mut f = 0.0;
                if ea_max - ea_min != 0.0 {
                    f = (y[t] - ea_min) / (ea_max - ea_min)
                }

                let (e1_min, e1_max) = DFO::get_energy_min_max(&dfo1.slices[t], s1, eps);
                y1.push(e1_min + f * (e1_max - e1_min));

                let (e2_min, e2_max) = DFO::get_energy_min_max(&dfo2.slices[t], s2, eps);
                y2.push(e2_min + f * (e2_max - e2_min));
            }
        }
        return (y1, y2)
    }

    pub fn disaggregate_1_n(dfo_a: DFO, y: Vec<f64>, dfos: &Vec<DFO>, eps: f64) -> Vec<Vec<f64>> {
        let mut dfo_a = dfo_a;
        let mut ys = Vec::with_capacity(dfos.len());
        let mut dummy_dfo = dfos.first().unwrap().clone();
        DFO::prepare_for_agg(&mut dummy_dfo, &mut dfo_a);
        for dfo in dfos {
            let mut dfo = dfo.clone();
            DFO::prepare_for_agg(&mut dfo, &mut dfo_a);
            let (_, y2) = DFO::disagg1to2(&dummy_dfo, &dfo, &dfo_a, &y, eps);
            ys.push(y2);
        }
        return ys;
    }

    fn from_chrg_events(events: Vec<ChrgEvent>,  eta: f64, delta_t: f64, num_samples: i32, eps: f64) -> Vec<DFO> {
        events.iter()
            .filter(|x| x.energy_departure > x.energy_arrival) // Filter parking events without charging
            .map(|x| DFO::inner(x, num_samples, eta, delta_t, eps)).collect()
    }
}

#[pyfunction]
pub fn gift_wrapping(points: Vec<(f64, f64)>, eps: f64)  -> Vec<(f64, f64)> {
    let mut convex_hull = Vec::new();
    let mut point_on_hull = *points.iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0)
        .unwrap())
        .unwrap();

    for _ in 0..points.len() {
        convex_hull.push(point_on_hull);
        let mut endpoint = points[0];

        for canditate in points.iter().skip(1) {
            let canditate = *canditate;
            if endpoint == point_on_hull {
                endpoint = canditate;
                continue;
            }

            // Line check
            let v = 
                (endpoint.0 - point_on_hull.0) * (canditate.1 - point_on_hull.1) -
                (endpoint.1 - point_on_hull.1) * (canditate.0 - point_on_hull.0)
            ;
            
            if v > eps {
                endpoint = canditate
            } else if v > -eps {
                // Compute distances
                let sqrd_endpnt = 
                    (endpoint.0 - point_on_hull.0).powi(2) +
                    (endpoint.1 - point_on_hull.1).powi(2)
                ;
                let sqrd_canditate = 
                    (canditate.0 - point_on_hull.0).powi(2) +
                    (canditate.1 - point_on_hull.1).powi(2)
                ;

                // Check if candiate and endpoint are identical to avoid infinite loops
                if f64::abs(sqrd_canditate - sqrd_endpnt) < eps {
                    if canditate == convex_hull[0] {
                        endpoint = canditate;
                    }
                // Else keep the furthest point from point_on_hull
                } else if sqrd_canditate > sqrd_endpnt {
                    endpoint = canditate;
                }
            }
        }
        point_on_hull = endpoint;

        if point_on_hull == convex_hull[0] {
            break;
        }
    }
    return convex_hull;
}

#[pyfunction]
pub fn aggregate_pipeline(
    events: Vec<ChrgEvent>, eta:f64, delta_t: f64, num_samples: i32, eps: f64,
    est: i32, lst: i32
) -> (Vec<DFO>, Vec<Vec<DFO>>) {
    let dfos = DFO::from_chrg_events(events, eta, delta_t, num_samples, eps);
    let (agg_dfos, groups) = DFO::aggregate_n_m(dfos, est, lst, num_samples, eps);
    let dfo_groups = groups.into_iter().map(|x| x.cells.into_iter().flat_map(|c| c.objects).collect()).collect();
    return (agg_dfos, dfo_groups);
}

#[pyfunction]
pub fn disaggregate_and_add(dfo_as: Vec<DFO>, ys: Vec<Vec<f64>>, groups: Vec<Vec<DFO>>, eta: f64, delta_t: f64, eps: f64) -> Vec<f64> {
    let mut rslt = vec![0.0; 672];
    for (dfo_a, y, group) in izip!(dfo_as, ys, groups) {
        let dfo_a_start = dfo_a.start;
        let ys_grp = DFO::disaggregate_1_n(dfo_a, y, &group, eps);

        for (y_dfo, dfo) in izip!(ys_grp, &group) {
            for (k, p) in y_dfo.iter().enumerate() {
                let start = i32::min(dfo.start, dfo_a_start) as usize;
                rslt[k + start] += p / delta_t / eta;
            }
        }
    }
    return rslt;
}
