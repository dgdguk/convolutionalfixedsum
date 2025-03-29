"""
figure_2
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from __future__ import annotations
from typing import Callable, Optional
from convolutionalfixedsum.itp import GOLDEN_RATIO, itp_max_iter
import math
import warnings
from convolutionalfixedsum import itp

# Monkey patch ITP to get a list of guesses out

ITP_GUESSES = []

def itp_get_guesses(func: Callable[[float], float], a: float, b: float, c: float = 0.0,
        epsilon: float = 1e-10, k1: Optional[float] = 0.2,
        k2: float = 0.99 * (1 + GOLDEN_RATIO), n0: int = 1,
        max_iter: Optional[bool, int] = None) -> float:
    """ITP (Interpolate, Truncate, Project) root finding method.
    Arguments:
        func: The function to search over
        a, b: The interval in which a f(x) = c lies
        c: Find intersections with line x=c; defaults to 0 (roots)
        epsilon: Required level of accuracy, default 1e-10
        k1, k2, n0: Hyperparameters of ITP method. Default values are derived
        from [1], and should be a) sane and b) lead to good performance.
        max_iter: If true, enforce the theoretical max number of iterations.
                  If a number, enforce this number of iterations.
    """
    global ITP_GUESSES
    # Comments referring to Olivera and Takahashi refer to Algorithm 1, in [1]
    if a == b: raise ValueError('a and b must not be equal')
    # Olivera and Takahashi assume a < b, so swap them if this is not true.
    if a > b:
        a, b = b, a
    y_a = func(a) - c
    y_b = func(b) - c
    if abs(y_a) <= epsilon:
        return a
    elif abs(y_b) <= epsilon:
        return b

    # Pseudocode in Olivera and Takahashi assumes y_a < 0 < y_b.
    # To ensure this, we'll take the sign of y_b and multiply everything we obtain from
    # func by it, creating a derived function that obeys the inequality.
    direction = math.copysign(1, y_b)
    y_a *= direction
    y_b *= direction

    if max_iter is True:
        max_iter = itp_max_iter(a, b, epsilon, n0)

    assert y_a < 0 and y_b > 0, f"func({a})={y_a} and func({b})={y_b} must be on opposite sides of zero"

    if k1 is None:
        k1 = 0.2 / (b - a)
    # Hyperparameter check
    assert k1 > 0, "Hyperparamter k1 must be positive"
    assert 1 <= k2 < 1 + GOLDEN_RATIO, "Hyperparameter k2 must be between 1 and 1 + 0.5*(1+math.sqrt(5))"
    assert n0 >= 0, "Hyperparameter n0 must be >= 0"
    if n0 == 0: warnings.warn('Setting n0 == 0 has the potential to cause numerical instability, '
                              'and this implementation of ITP does not check against this')

    n_half = math.ceil(math.log2((b - a) / (2 * epsilon)))
    n_max = n_half + n0
    k = 0

    while (b - a > 2 * epsilon):
        # Interpolate
        x_f = (y_b * a - y_a * b) / (y_b - y_a)
        # Truncate
        x_half = (a + b) / 2
        sigma = math.copysign(1, x_half - x_f)
        delta = k1 * ((b - a) ** k2)
        if delta <= abs(x_half - x_f):
            x_t = x_f + math.copysign(delta, sigma)
        else:
            x_t = x_half
        # Projection, equation(15)
        r = epsilon * (2 ** (n_max - k)) - (b - a) / 2
        if abs(x_t - x_half) <= r:
            x_itp = x_t
        else:
            x_itp = x_half - math.copysign(r, sigma)
        # Update interval
        ITP_GUESSES.append(x_itp)
        y_itp = (func(x_itp) - c) * direction
        if y_itp > 0:
            b = x_itp
            y_b = y_itp
        elif y_itp < 0:
            a = x_itp
            y_a = y_itp
        else:
            a = b = x_itp
        if max_iter is not None and k > (max_iter):
            raise Exception(
                f'Non-convergence detected (precision {abs(a - b)}); is your function precise enough? (max_iter={max_iter})')
        k += 1
    result = (a + b) / 2
    ITP_GUESSES.append(result)
    return result

itp.itp = itp_get_guesses

from convolutionalfixedsum.cfsvr import CFSVR
import pyvista as pv

from viz import add_axes, add_2simplex, hyperplane_mesh, add_line_on_hyperplane_simplex_constraint

def add_region(pl, x1, x2, simplex, color):
    mesh1, mesh2 = hyperplane_mesh(2, x1, x2, simplex)

    pl.add_mesh(mesh1, color=color, ambient=0.25)
    pl.add_mesh(mesh2, color=color, ambient=0.25)

COLOR_LC = 'green'
COLOR_UC = 'red'
COLOR_REGION =  'gold'
COLOR_ITP_GUESS = [0, 128, 255, 128]
COLOR_ITP_ANSWER = [0, 0, 255, 255]
COLOR_SELECTED_POINT = [255, 0, 255]


def plot_cfs_example_graph(constraints, first_variate_i, second_variate_i, show_legend=True, itp_guesses=False):
    # Note: Due to the Z axis being horizontal, and wanting to use that as the visualised axis,
    # the constraints are solved in the order 3-2-1 (or Z, Y, X).

    # Use the numeric version, as the analytical version is in C and therefore uses the C implementation of ITP.
    # C is obviously more resistant to monkey patching.
    dist = CFSVR([constraints[2], constraints[1], constraints[0]], 1.0, 10000)
    first_variate_s = dist.inverse_cdf(first_variate_i)

    pl = pv.Plotter()
    add_axes(pl)
    add_2simplex(pl, [0, 0, 0], 1.0, COLOR_LC)
    add_2simplex(pl, constraints, 1.0, COLOR_UC)

    intersections = [1 - x for x in constraints]
    region_sections = [0]
    if first_variate_s > intersections[0]: region_sections.append(intersections[0])
    if first_variate_s > intersections[1]: region_sections.append(intersections[1])
    region_sections.append(first_variate_s)

    for x1, x2 in zip(region_sections, region_sections[1:]):
        add_region(pl, x1, x2, constraints, COLOR_REGION)

    if itp_guesses:
        for x in range(0, 3):
            add_line_on_hyperplane_simplex_constraint(pl, 2, ITP_GUESSES[x], constraints, COLOR_ITP_GUESS, 5)
    add_line_on_hyperplane_simplex_constraint(pl, 2, first_variate_s, constraints, COLOR_ITP_ANSWER, 7)

    remaining_alloc = 1.0 - first_variate_s
    # Caculate the upper / lower bound on penultimate task
    min_second_variate_s = max(0, remaining_alloc - constraints[0])
    max_second_variate_s = min(remaining_alloc, constraints[1])

    # Pick the value at second_variate_i between them
    second_variate_s = second_variate_i * (max_second_variate_s - min_second_variate_s) + min_second_variate_s

    third_variate_s = 1.0 - first_variate_s - second_variate_s

    sphere = pv.Sphere(radius=0.03, center=[third_variate_s, second_variate_s, first_variate_s])
    pl.add_mesh(sphere, color=COLOR_SELECTED_POINT, ambient=1.0, label='Selected Point')
    #point = pv.PointSet([[third_variate_s, second_variate_s, first_variate_s]])
    #pl.add_mesh(point, color=[0, 255, 0], point_size=20, render_points_as_spheres=True, ambient=1.0)
    print(f'Constraints given: {constraints}')
    print(f'Found point: {[third_variate_s, second_variate_s, first_variate_s]}')
    print(f'First 3 ITP guesses: {ITP_GUESSES[0:3]}')
    print(f'Min/max 2nd variate: {min_second_variate_s}, {max_second_variate_s}')
    pl.show_grid()
    if show_legend:
        legend = [[f'  Region covering {round(first_variate_i, 3)} \n  of valid region', COLOR_REGION, 'r']]
        if itp_guesses:
            legend.append([f'  ITP Root Finding', COLOR_ITP_GUESS, 'r'])
        legend.append([f'  ITP Solution z={round(first_variate_s, 3)}', COLOR_ITP_ANSWER, 'r'])
        legend.append([f'  Solution splitting line \n  z={round(first_variate_s, 3)} at {round(second_variate_i, 3)}',
                       COLOR_SELECTED_POINT, 'r'])
        pl.add_legend(legend, size=[0.3, 0.3])
    pl.show()
    pl.screenshot('figure_2.png')

if __name__ == '__main__':
    constraints = [0.5, 0.7, 0.8]  # Values for the constraints
    # variate_i values represent the values of calls to random() in the normal algorithm.
    # If you change these both to random(), you'll generate uniform points.
    first_variate_i = 0.52         # Value to divide Z around i.e. first_variate_i * total_volume lies below the divide
    second_variate_i = 0.12        # Value to divide X-Y around

    show_legend=True   # Show a legend. Due to a lack of ability to control font color, for the paper this was disabled
                       # and a legend added by hand.
    itp_guesses=False  # Show the first 3 guesses for ITP along the Z axis. This was disabled for the paper as it
                       # rendered the graph too busy, but can be informative.
    plot_cfs_example_graph(constraints, first_variate_i, second_variate_i, show_legend, itp_guesses)


