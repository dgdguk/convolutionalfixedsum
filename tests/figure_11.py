"""
figure_11
*********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import pyvista as pv
from .viz import do_csvr_graph
from .drs_uniformity import drs_uniformity_test
import random

if __name__ == '__main__':
    random.seed(1)
    pl = pv.Plotter()
    constraints = [1, 1, 1/4, 1e-4]
    data = drs_uniformity_test(constraints, runs=10000)
    do_csvr_graph(pl, constraints[0:3], data, 2, palette='kbc', vertical_map=False)
    pl.show_grid()
    pl.show()
    pl.screenshot('figure_11.png')
