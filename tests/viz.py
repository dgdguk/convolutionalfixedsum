"""
viz
***

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import pyvista as pv
import numpy as np
import colorcet as cc

def cts(limits: np.ndarray, total: float) -> np.ndarray:
    """Converts constraints into the coodinates of a constraints simplex"""
    simplex_coords = np.zeros([limits.size, limits.size])
    for index in range(len(limits)):
        simplex_coords[index][:] = limits
        simplex_coords[index][index] = 0.0
        simplex_coords[index][index] = total - simplex_coords[index].sum()
    return simplex_coords
    
def add_axes(pl, length=1.2):
    axes = np.zeros([6, 3])
    for ax in range(3):
        axes[2*ax+1][ax] = 1.0
    pl.add_lines(axes, 'black', 5)
    
def add_line_on_hyperplane(pl, dimension, x):
    off_dimensions = [0, 1, 2]
    off_dimensions.remove(dimension)
    lines = np.zeros([2, 3])
    lines[0, dimension] = x
    lines[1, dimension] = x
    lines[0, off_dimensions[0]] = 1 - x
    lines[1, off_dimensions[1]] = 1 - x
    pl.add_lines(lines, 'green', 5)

def add_line_on_hyperplane_simplex_constraint(pl, dimension, x, simplex, color, thickness):
    off_dimensions = [0, 1, 2]
    off_dimensions.remove(dimension)
    lines = np.zeros([2, 3])
    lines[0, dimension] = x
    lines[1, dimension] = x
    lines[0, off_dimensions[0]] = max(1.0 - x - simplex[off_dimensions[1]], 0.0)
    lines[1, off_dimensions[1]] = max(1.0 - simplex[off_dimensions[0]] - x, 0.0)

    lines[0, off_dimensions[1]] = 1.0 - x - lines[0, off_dimensions[0]]
    lines[1, off_dimensions[0]] = 1.0 - x - lines[1, off_dimensions[1]]
    pl.add_lines(lines, color, thickness)

def hyperplane_mesh(dimension, x1, x2, simplex):
    off_dimensions = [0, 1, 2]
    off_dimensions.remove(dimension)

    points = np.zeros([4, 3])
    points[0, dimension] = x1
    points[1, dimension] = x1

    points[0, off_dimensions[0]] = max(1.0 - x1 - simplex[off_dimensions[1]], 0.0)
    points[1, off_dimensions[1]] = max(1.0 - simplex[off_dimensions[0]] - x1, 0.0)
    points[0, off_dimensions[1]] = 1.0 - x1 - points[0, off_dimensions[0]] #simplex[off_dimensions[1]]
    points[1, off_dimensions[0]] = 1.0 - x1 - points[1, off_dimensions[1]] #simplex[off_dimensions[0]]
    points[2, dimension] = x2
    points[3, dimension] = x2
    points[2, off_dimensions[0]] = max(1.0 - x2 - simplex[off_dimensions[1]], 0.0)
    points[3, off_dimensions[1]] = max(1.0 - simplex[off_dimensions[0]] - x2, 0.0)
    points[2, off_dimensions[1]] = 1.0 - x2 - points[2, off_dimensions[0]] #simplex[off_dimensions[1]]
    points[3, off_dimensions[0]] = 1.0 - x2 - points[3, off_dimensions[1]] #simplex[off_dimensions[0]]
    #print(points)
    mesh1 = pv.Triangle([points[0], points[1], points[2]])
    mesh2 = pv.Triangle([points[1], points[2], points[3]])
    return mesh1, mesh2

def add_region_on_hyperplane_simplex_constraint(pl, dimension, x1, x2, simplex, value, palette='fire', vertical_map=True):
    # TODO: Handle UC tip inside LC simplex which results in a single triangle at tip of UC simplex
    mesh1, mesh2 = hyperplane_mesh(dimension, x1, x2, simplex)
    values = np.zeros(3)
    values = value
    mesh1['values'] = values
    mesh2['values'] = values
    scalar_bar_args = {'title': "Normalised Point Density", 'title_font_size': 26, 'label_font_size': 24}
    if vertical_map:
        scalar_bar_args.update(
            {'title': "Normalised\nPoint\nDensity\n\n", 'vertical': True, 'height': 0.9, 'position_x': 0.85}
        )
    pl.add_mesh(mesh1, style='surface', cmap=palette, scalar_bar_args=scalar_bar_args)
    pl.add_mesh(mesh2, style='surface', cmap=palette, scalar_bar_args=scalar_bar_args)


def add_2simplex(pl, limits, total, color):
    limits = np.asarray(limits)
    simplex_coords = cts(limits, total)
    lines = np.zeros([len(limits) * 2, 3])
    for x in range(len(limits)):
        lines[2*x][:] = simplex_coords[x]
        lines[2*x+1][:] = simplex_coords[(x + 1) % len(limits)]
    pl.add_lines(lines, color, 6, connected=True)

def do_csvr_graph(pl, constraints, data, dim, palette='fire', vertical_map=True):
    add_axes(pl)
    add_2simplex(pl, [0, 0, 0], 1, 'green')
    if constraints:
        add_2simplex(pl, constraints, 1, 'red')
    else:
        constraints = [1, 1, 1]
    values = data[0][dim]
    total = sum(values)
    values = [x * len(values) / total for x in values]
    brackets = [x for x in zip(data[-1][dim], data[-1][dim][1:])]
    for value, bracket in zip(values, brackets):
        add_region_on_hyperplane_simplex_constraint(pl, 2, bracket[0], bracket[1], constraints, value, palette, vertical_map)
