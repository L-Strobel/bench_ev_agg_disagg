use itertools::izip;
use pyo3::prelude::*;

use crate::fo_grouping::Group;
use crate::{chrg_event::ChrgEvent, fo_grouping as fogrp};

#[pyclass]
#[derive(Copy, Clone)]
pub struct Slice {
    amin: f64,
    amax: f64
}

#[pyclass]
pub struct FO {
    #[pyo3(get)]
    t_earliest: i32,
    #[pyo3(get)]
    t_latest: i32,
    #[pyo3(get)]
    ammount_profile: Vec<Slice>,
    // For instantiated FOs:
    #[pyo3(get)]
    t_realized: Option<i32>,
    #[pyo3(get)]
    profile_realized: Option<Vec<f64>>
}

impl FO {
    fn time_flex(self: &FO) -> i32 {
        self.t_latest - self.t_earliest
    }

    fn aggregate_n_1(fos: Vec<&FO>) -> FO {
        // Determine earliest and latest start time
        let t_earliest = fos.iter().map(|x| x.t_earliest).min().unwrap();
        let min_t_flex = fos.iter().map(|x| x.time_flex()).min().unwrap();
        let t_latest = t_earliest + min_t_flex;

        // Create Profile
        let latest_time = fos.iter().map(|x| x.t_earliest + (x.ammount_profile.len() as i32)).max().unwrap();
        let profile_len = (latest_time - t_earliest) as usize;
        let mut ammount_profile = vec![Slice{amin: 0.0, amax: 0.0}; profile_len];
        for fo in fos {
            let offset = (fo.t_earliest - t_earliest) as usize;
            for (t, s) in fo.ammount_profile.iter().enumerate() {
                ammount_profile[t+offset].amin += s.amin;
                ammount_profile[t+offset].amax += s.amax;
            }
        }
        return FO{t_earliest, t_latest, ammount_profile, t_realized: None, profile_realized: None}
    }

    fn aggregate_n_m(fos: Vec<FO>, est: i32, tft: i32) -> (Vec<FO>, Vec<Group<FO>>) {
        fn get_x(fo: &FO) -> i32 { fo.t_earliest }
        let groups = fogrp::pre_group(fos, est, tft, get_x, FO::time_flex);
        let opt_groups = fogrp::optimize_groups(groups, est, tft, get_x, FO::time_flex);

        let mut agg_fos = Vec::with_capacity(opt_groups.len());
        for group in &opt_groups {
            let group_fos: Vec<&FO> = group.cells.iter().flat_map(|x| x.objects.iter()).collect(); 
            agg_fos.push(FO::aggregate_n_1(group_fos));
        }
        return (agg_fos, opt_groups);
    }

    fn disaggregate_1_n(fos: Vec<FO>, fo_a: &FO, alignments: Vec<i32>) -> Vec<FO> {
        let mut fos = fos;
        let a_realized_profile = fo_a.profile_realized.as_ref().unwrap();
        let relative_profile: Vec<f64> = izip!(&fo_a.ammount_profile, a_realized_profile)
            .map(|(s, sx)| {
                if s.amax == s.amin {
                    1.0
                } else {
                    (sx - s.amin) / (s.amax - s.amin)
                }
            }).collect();
        
        let shift = fo_a.t_realized.unwrap() - fo_a.t_earliest;
        for (i, fo) in fos.iter_mut().enumerate() {
            let alignment = alignments[i];
            fo.t_realized = Some(shift + alignment);

            let offset = (alignment - fo_a.t_earliest) as usize;
            let mut profile_realized = Vec::with_capacity(fo.ammount_profile.len());
            for (j, s) in fo.ammount_profile.iter().enumerate() {
                profile_realized.push(
                    s.amin + (s.amax - s.amin) * relative_profile[offset + j]
                )
            }
            fo.profile_realized = Some(profile_realized);
        }
        return fos
    }

    fn from_chrg_events(events: &Vec<ChrgEvent>, eta: f64, delta_t: f64) -> Vec<FO> {
        let mut fos = Vec::new();
        for event in events {
            if event.energy_arrival >= event.energy_departure { continue; }

            // Determine ammount profile
            let mut ammount_profile = Vec::new();
            let mut soe = event.energy_arrival;
            for _ in event.start..event.stop {
                let e_charge = f64::max(0.0, f64::min(event.p_max * eta * delta_t, event.energy_departure - soe));
                soe += e_charge;

                if e_charge <= 0.0 { break; }
                ammount_profile.push( Slice{amin: e_charge / eta, amax: e_charge / eta} ); 
            }

            let fo = FO {
                t_earliest: event.start, t_latest: event.stop - (ammount_profile.len() as i32), ammount_profile,
                t_realized: None, profile_realized: None};
            fos.push(fo);
        }
        return fos;
    }

    fn optimize(fos: Vec<FO>, costs: &Vec<f64>) -> Vec<FO> {
        let mut fos = fos;
        for fo in &mut fos {
            let mut z_best = f64::INFINITY;
            let mut t_best = 0;

            for t in fo.t_earliest..(fo.t_latest+1) {
                let mut z = 0.0;
                for (i, s) in fo.ammount_profile.iter().enumerate() {
                    z += s.amin * costs[t as usize + i]
                }
                if z < z_best {
                    z_best = z;
                    t_best = t;
                }
            }
            fo.t_realized = Some(t_best);
            fo.profile_realized = Some(fo.ammount_profile.iter().map(|x| x.amin).collect());
        }
        return fos;
    }
}

#[pyfunction]
pub fn pipeline(events: Vec<ChrgEvent>, eta: f64, delta_t: f64, costs: Vec<f64>, est: i32, tft: i32) -> (Vec<f64>, usize) {
    // Create
    let fos = FO::from_chrg_events(&events, eta, delta_t);

    // Aggregate
    let (agg_objects, groups) = FO::aggregate_n_m(fos, est, tft);

    // Optimize
    let agg_objects = FO::optimize(agg_objects, &costs);

    // Disaggregate
    let mut instances = Vec::new();
    for (fo_a, group) in izip!(&agg_objects, groups) {
        let grp_fos: Vec<FO> = group.cells.into_iter().flat_map(|x| x.objects.into_iter()).collect();
        let alignments = grp_fos.iter().map(|x| x.t_earliest).collect();
        let mut grp_fos_i = FO::disaggregate_1_n(grp_fos, fo_a, alignments);
        instances.append(&mut grp_fos_i);
    }

    // Total power
    let mut rslt = vec![0.0; costs.len()];
    for fo in instances {
        let offset = fo.t_realized.unwrap() as usize;
        for (i, p) in fo.profile_realized.unwrap().iter().enumerate() {
            rslt[i+offset] += p / delta_t;
        }
    }
    return  (rslt, agg_objects.len());
}