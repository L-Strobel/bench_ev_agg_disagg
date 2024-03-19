"""
Grouping algorithm for FOs.
Based on: https://doi.org/10.1109/TKDE.2015.2445755
"""
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

@dataclass
class Cell:
    """
    Cell of grouping grid
    """
    x_min: int
    y_min: int
    objects: list[object]
    x_centroid: float = None
    y_centroid: float = None

@dataclass
class Group:
    """
    Group.
    """
    cells: list[Cell]

def pre_group(
        objects: object, x_thresh: float, y_thresh: float, get_x: Callable, get_y: Callable
    ) -> list[Group]:
    """
    Pregrouping. Create unoptimized groups.
    :param objects: Objects to group
    :param x_thresh: Threshhold for first attribute
    :param y_thresh: Threshhold for second attribute
    :param get_x: get_x(object) -> first attribute
    :param get_y: get_y(object) -> second attribute
    """
    # Get populated cells
    populated_cells = {}
    for o in objects:
        key = (get_x(o) // (x_thresh+1), get_y(o) // (y_thresh+1))
        if key in populated_cells:
            populated_cells[key].objects.append(o)
        else:
            populated_cells[key] = Cell(get_x(o), get_y(o), [o])
    # Combine adjacent groups
    group_hash = {}
    groups = []
    for base_key, cell in populated_cells.items():
        # Go through adjacent cells
        for i, j in [(-1, 0), (1,0), (0,0), (0, 1), (0, -1)]:
            key = (base_key[0] + i, base_key[1] + j)
            if key in group_hash:
                group = group_hash[key]
                group_hash[base_key] = group
                group.cells.append(cell)
                break
        else:
            new_group = Group([cell])
            group_hash[base_key] = new_group
            groups.append(new_group)
    return groups

def get_cell_centroid(cell: Cell, get_x: Callable, get_y: Callable ) -> tuple[float, float]:
    """
    Get centeroid of a cell.
    :param cell: Cell
    :param get_x: get_x(object) -> first attribute
    :param get_y: get_y(object) -> second attribute
    """
    objects = [o for o in cell.objects]
    x_min = get_x( min(objects, key=get_x) )
    x_max = get_x( max(objects, key=get_x) )
    y_min = get_y( min(objects, key=get_y) )
    y_max = get_y( max(objects, key=get_y) )
    return x_max - x_min, y_max - y_min

def is_mbr_ok(
        group: Group, x_thresh: float, y_thresh: float, get_x: Callable, get_y: Callable
    ) -> bool:
    """
    Check if minimal bouding recantangle of the group does not violates the threshholds.
    :param group: Group
    :param x_thresh: Threshhold for first attribute
    :param y_thresh: Threshhold for second attribute
    :param get_x: get_x(object) -> first attribute
    :param get_y: get_y(object) -> second attribute
    """
    objects = [o for c in group.cells for o in c.objects]
    # x-Coordinate
    te_min = get_x( min(objects, key=get_x) )
    te_max = get_x( max(objects, key=get_x) )
    # y-Coordinate
    tf_min = get_y( min(objects, key=get_y) )
    tf_max = get_y( max(objects, key=get_y) )
    # Check
    te_ok = (te_max - te_min) <= x_thresh
    tf_ok = (tf_max - tf_min) <= y_thresh
    return te_ok and tf_ok

def cluster_distance(group_a, group_b):
    """
    Maximum linkage
    """
    max_distance = 0
    for c_a in group_a.cells:
        for c_b in group_b.cells:
            distance = math.sqrt(
                (c_a.x_centroid-c_b.x_centroid)**2 + (c_a.y_centroid-c_b.y_centroid)**2
            )
            if distance > max_distance:
                max_distance = distance
    return max_distance

def cluster_hierarch(
        cells: list[Cell], x_thresh: float, y_thresh: float, get_x: Callable, get_y: Callable
    ) -> list[Group]:
    """
    Cluster the cells into groups.
    :param cells: Cells
    :param x_thresh: Threshhold for first attribute
    :param y_thresh: Threshhold for second attribute
    :param get_x: get_x(object) -> first attribute
    :param get_y: get_y(object) -> second attribute
    """
    groups = [Group([c]) for c in cells]
    for c in cells:
        x, y = get_cell_centroid(c, get_x, get_y)
        c.x_centroid = x
        c.y_centroid = y
    while len(groups) > 1:
        # Find minimum distance between to groups
        min_pair = None
        min_distance = np.inf
        n = len(groups)
        for i in range(n):
            for j in range(i+1, n):
                distance = cluster_distance(groups[i], groups[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)
        # Merge groups
        (i, j) = min_pair
        group_a = groups[i]
        group_b = groups[j]
        new_group = Group(group_a.cells + group_b.cells)
        if is_mbr_ok(new_group, x_thresh, y_thresh, get_x, get_y):
            groups.remove(group_a)
            groups.remove(group_b)
            groups.append(new_group)
        else:
            break
    return groups

def optimize_groups(
        groups: list[Group],  x_thresh: float, y_thresh: float, get_x: Callable, get_y: Callable
    ) -> list[Group]:
    """
    Optimize grouping. Only splitting needs to be done.
    All adjacent groups are already merged in pre_group().
    :param groups: Groups
    :param x_thresh: Threshhold for first attribute
    :param y_thresh: Threshhold for second attribute
    :param get_x: get_x(object) -> first attribute
    :param get_y: get_y(object) -> second attribute
    """
    opt_groups = []
    for group in groups:
        if not is_mbr_ok(group, x_thresh, y_thresh, get_x, get_y):
            opt_groups.extend(cluster_hierarch(group.cells, x_thresh, y_thresh, get_x, get_y))
        else:
            opt_groups.append(group)
    return opt_groups
