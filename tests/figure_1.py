"""
figure_1
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from __future__ import annotations
import pyvista as pv
from collections import Counter
import numpy as np

def add_axes(pl, length=1.2, ax_start=0, ax_stop=3, zero_point=(0.0, 0.0, 0.0)):
    ax_labels = {0: 'x', 1: 'y', 2: 'z'}
    label_offsets = {0: np.array([0.3, 0.0, 0.0]), 1:np.array([0.0, 0.1, 0.0]), 2:np.array([0.0, 0.0, 0.0])}
    zero_point = np.asarray(zero_point)
    for ax in range(ax_start, ax_stop):
        direction = [0.0] * 3
        direction[ax] = length
        arrow = pv.Arrow(start=zero_point, direction=direction, tip_length= 0.25 / length,
                         tip_radius = 0.1 / length, shaft_radius=0.05/ length, scale=length)
        pl.add_mesh(arrow, color='black')
        label = pv.Label(ax_labels[ax], position=direction + label_offsets[ax] + zero_point, size=35)
        pl.add_actor(label)


CUBE_CENTERS = [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]

COLOR_LC = [40, 40, 180]
COLOR_UC = 'orange'

AMBIENT_LC = 0.4
AMBIENT_UC = 0.4

def cubular_icdf(cubes, target, variate_i):
    cubes_per_index = Counter(cube[target] for cube in cubes)
    target_variate_i = len(cubes) * variate_i
    total_alloc = 0.0
    index = 0
    for index in sorted(cubes_per_index):
        if total_alloc + cubes_per_index[index] > target_variate_i: break
        else: total_alloc += cubes_per_index[index]

    remaining_alloc = target_variate_i - total_alloc  # I need this many full cubes
    cubes_on_index = cubes_per_index[index] # And there are this many cubes on the index
    ret = index + (remaining_alloc / cubes_on_index)

    return ret

def cube_shift(cube, target, zero_val):
    cube = cube[:]
    cube[0] = (cube[0] + zero_val[0] + 0.5) if target < 1 else (zero_val[0] + 0.05)
    cube[1] = (cube[1] + zero_val[1] + 0.5) if target < 2 else (zero_val[1] + 0.05)
    cube[2] += 0.5
    return cube

def cube_gen(pl, cubes, target, variate_s, zero_point=(0.0, 0.0, 0.0)):
    mp = {0: 'x_length', 1: 'y_length', 2: 'z_length'}
    default_size = dict(zip(['x_length', 'y_length', 'z_length'], [1 if x >= target else 0.1 for x in range(3)]))
    for cube_center in cubes:
        if cube_center[target] < int(variate_s):
            # This is below our slice
            mesh = pv.Cube(cube_shift(cube_center, target, zero_point), **default_size)
            pl.add_mesh(mesh, color=COLOR_LC, ambient=AMBIENT_LC)
        elif cube_center[target] > int(variate_s):
            # This is above our slice
            mesh = pv.Cube(cube_shift(cube_center, target, zero_point), **default_size)
            pl.add_mesh(mesh, color=COLOR_UC, ambient=AMBIENT_UC)
        else:
            # We need to split this into two
            cube_1_center = cube_center[:]
            cube_1_center[target] = (int(variate_s) - 1 + variate_s) / 2
            cube_1_size = default_size.copy()
            cube_1_size[mp[target]] = variate_s % 1
            mesh = pv.Cube(cube_shift(cube_1_center, target, zero_point), **cube_1_size)
            pl.add_mesh(mesh, color=COLOR_LC, ambient=AMBIENT_LC)

            cube_2_center = cube_center[:]
            cube_2_center[target] = (int(variate_s) + variate_s) / 2
            cube_2_size = default_size.copy()
            cube_2_size[mp[target]] = 1 - variate_s % 1
            mesh = pv.Cube(cube_shift(cube_2_center, target, zero_point), **cube_2_size)
            pl.add_mesh(mesh, color=COLOR_UC, ambient=AMBIENT_UC)

def plot_cube_example(cubes, first_variate_i, second_variate_i, third_variate_i, offset=None):
    assert len(cubes) > 0

    for cube in cubes:
        if not all(isinstance(x, int) and x >= 0 for x in cube):
            raise ValueError('All cube center coordinates supplied should be non-negative integers')

    if offset is None:
        # Guess a decent offset between graphs.
        offset = max(max(x) for x in cubes) + 0.5

    first_variate_s = cubular_icdf(cubes, 0, first_variate_i)
    cubes_v2 = sorted([cube[:] for cube in cubes if cube[0] == int(first_variate_s)], key=lambda cube: cube[1])
    second_variate_s = cubular_icdf(cubes_v2, 1, second_variate_i)
    cubes_v3 = sorted([cube[:] for cube in cubes_v2 if cube[1] == int(second_variate_s)], key=lambda cube: cube[2])
    third_variate_s = cubular_icdf(cubes_v3, 2, third_variate_i)

    graph1_zero = np.array([0, 0, 0])
    graph2_zero = np.array([-offset, offset, 0])
    graph3_zero = np.array([-2*offset, 2*offset, 0])

    pl = pv.Plotter()
    add_axes(pl, 5, 0, 3, graph1_zero)
    add_axes(pl, 5, 1, 3, graph2_zero)
    add_axes(pl, 5, 2, 3, graph3_zero)

    cube_gen(pl, cubes, 0, first_variate_s, graph1_zero)
    cube_gen(pl, cubes_v2, 1, second_variate_s, graph2_zero)
    cube_gen(pl, cubes_v3, 2, third_variate_s, graph3_zero)

    label = pv.Label(f"{round(first_variate_i, 3)} split\nat x={round(first_variate_s, 2)}",
                     position=graph1_zero + np.array([5, 4, 0]), size=30)
    pl.add_actor(label)

    label = pv.Label(f"{round(second_variate_i, 3)} split\nat y={round(second_variate_s, 2)}",
                     position=graph2_zero + np.array([5, 4, 0]), size=30)
    pl.add_actor(label)

    label = pv.Label(f"{round(third_variate_i, 3)} split\nat z={round(third_variate_s, 2)}",
                     position=graph3_zero + np.array([5, 4, 0]), size=30)
    pl.add_actor(label)
    print(f'Selected Point = {first_variate_s, second_variate_s, third_variate_s}')
    pl.show()
    pl.screenshot('figure_1.png')


if __name__ == '__main__':
    # A list giving the coordinates for cube centers - all coordinates should be non-negative integers, but
    # otherwise you can change this however you like
    cube_centers = (
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 0, 1], [0, 0, 2], [1, 0, 1]
    )

    # For example, this is a smiley face. It's not as useful of an example, but illustrative.
    smiley_cube_centers = (
        [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 0, 1], [0, 4, 1], # Smile
        [0, 2, 2], [1, 2, 2], [0, 2, 3],  # Nose
        [0, 1, 5], [0, 1, 6], [0, 3, 5], [0, 3, 6] # Eyes
    )

    # variate_i values represent the result of a call to random() in the IVoRS algorithm for each dimension.
    # If you replace these with a call to random(), then you'll select a value with a uniform distribution
    first_variate_i = 0.72     # x coord
    second_variate_i = 0.45    # y coord
    third_variate_i = 0.31     # z coord

    plot_cube_example(cube_centers, first_variate_i, second_variate_i, third_variate_i)


