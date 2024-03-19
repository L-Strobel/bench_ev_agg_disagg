use std::collections::HashMap;

use indexmap::IndexMap;

#[derive(Clone)]
pub struct Cell<T> {
    pub objects: Vec<T>,
    x_centroid: i32,
    y_centroid: i32
}

#[derive(Clone)]
pub struct Group<T> {
    pub cells: Vec<Cell<T>>
}

impl <T> Cell<T> {
    fn update_centroid(&mut self, get_x: fn(&T) -> i32, get_y: fn(&T) -> i32) {
        let xmin = self.objects.iter().map(get_x).min().unwrap();
        let xmax = self.objects.iter().map(get_x).max().unwrap();
        let ymin = self.objects.iter().map(get_y).min().unwrap();
        let ymax = self.objects.iter().map(get_y).max().unwrap();
        self.x_centroid = xmax - xmin;
        self.y_centroid = ymax - ymin;
    }    
}

pub fn pre_group<T>(objects: Vec<T>, x_thresh: i32, y_thresh: i32, get_x: fn(&T) -> i32, get_y: fn(&T) -> i32) -> Vec<Group<T>> {
    // Get populated cells
    let mut populated_cells: IndexMap<(i32, i32), Cell<T>> = indexmap::IndexMap::new(); // Use indexmap for consistency with Python version.
    for o in objects {
        let x_o = get_x(&o);
        let y_o = get_y(&o);
        let key = (x_o / (x_thresh + 1), y_o / (y_thresh + 1));
        if populated_cells.contains_key(&key) {
            populated_cells.get_mut(&key).unwrap().objects.push(o);
        } else {
            let new_cell = Cell {objects: vec![o], x_centroid: -1, y_centroid: -1};
            populated_cells.insert(key, new_cell);
        }
    }
    // Combine adjacent groups
    let mut combined_groups: Vec<Group<T>> = Vec::new();
    let mut group_hash: HashMap<(i32, i32), usize> = HashMap::new();
    let key_shifts = [(-1, 0), (1,0), (0,0), (0, 1), (0, -1)];
    for (base_key, cell) in populated_cells {
        // Go through adjacent cells
        let used_key = key_shifts.iter()
            .map(|(i, j)| (base_key.0 + i, base_key.1 + j))
            .find(|x| group_hash.contains_key(&x));
        match used_key {
            Some(key) => {
                let pntr = *group_hash.get(&key).unwrap();
                group_hash.insert(base_key, pntr);
                combined_groups[pntr].cells.push(cell);
            }
            None => {
                let new_group = Group{cells: vec![cell]};
                let pntr = combined_groups.len();
                combined_groups.push(new_group);
                group_hash.insert(base_key, pntr);
            }
        }
    }
    return combined_groups;
}

fn mbr_ok<'a, I, T>(vals: I, x_thresh: i32, y_thresh: i32, get_x: fn(&T) -> i32, get_y: fn(&T) -> i32) -> bool 
where
    I: Iterator<Item = &'a Cell<T>>, I: Clone, T: 'a
{
    let xmin = vals.clone().flat_map(|x| &x.objects).map(get_x).min().unwrap();
    let xmax = vals.clone().flat_map(|x| &x.objects).map(get_x).max().unwrap();
    let ymin = vals.clone().flat_map(|x| &x.objects).map(get_y).min().unwrap();
    let ymax = vals.clone().flat_map(|x| &x.objects).map(get_y).max().unwrap();
    return (xmax - xmin <= x_thresh) && (ymax - ymin <= y_thresh)
}

fn maximum_linkage_distance<T>(grp_a: &Group<T>, grp_b: &Group<T>) -> f64 {
    let mut max_distance = 0.0;
    for c_a in &grp_a.cells {
        for c_b in &grp_b.cells {
            let distance = f64::sqrt(
                ((c_a.x_centroid - c_b.x_centroid).pow(2) + (c_a.y_centroid - c_b.y_centroid).pow(2)) as f64
            );
            max_distance = f64::max(distance, max_distance);
        }
    }
    return max_distance
}

fn hierachical_clustering<T>(cells: Vec<Cell<T>>, x_thresh: i32, y_thresh: i32, get_x: fn(&T) -> i32, get_y: fn(&T) -> i32) -> Vec<Group<T>> {
    let mut cells = cells;
    for c in &mut cells {
        c.update_centroid(get_x, get_y);
    }
    let mut groups: Vec<Group<T>> = cells.into_iter().map(|x| Group{cells: vec![x]}).collect();
    while groups.len() > 1 {
        // Find closest pair
        let mut min_pair: Option<(usize, usize)> = None;
        let mut min_distance = f64::INFINITY;
        for i in 0..groups.len() {
            for j in i+1..groups.len() {
                let distance = maximum_linkage_distance(&groups[i], &groups[j]);
                if distance < min_distance {
                    min_distance = distance;
                    min_pair = Some((i, j));
                }
            }
        }
        // Merge pair
        let (i, j) = min_pair.unwrap();
        let cell_iter = groups[i].cells.iter().chain(groups[j].cells.iter());
        if mbr_ok(cell_iter, x_thresh, y_thresh, get_x, get_y) {
            let group_a = groups.remove(j); // j > i
            let group_b = groups.remove(i);
            let mut cells = group_a.cells;
            cells.extend(group_b.cells);
            let new_group = Group{ cells };
            groups.push(new_group);
        } else {
            break;
        }
    }
    return groups
}

pub fn optimize_groups<T>(groups: Vec<Group<T>>, x_thresh: i32, y_thresh: i32, get_x: fn(&T) -> i32, get_y: fn(&T) -> i32) -> Vec<Group<T>> {
    let mut opt_groups = Vec::new();
    for group in groups {
        if !mbr_ok(group.cells.iter(), x_thresh, y_thresh, get_x, get_y) {
            let clustered = hierachical_clustering(group.cells, x_thresh, y_thresh, get_x, get_y);
            opt_groups.extend( clustered );
        } else {
            opt_groups.push(group)
        }
    }
    return opt_groups
}  