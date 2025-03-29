"""
figure_4
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import pyvista as pv
from .viz import do_csvr_graph
from .cfsvr_uniformity import cfsvr_uniformity_test
import random

if __name__ == '__main__':
    random.seed(1)
    pl = pv.Plotter()
    constraints = [0.7, 0.3, 0.5]
    data = cfsvr_uniformity_test(constraints, runs=10000, sample_signal_size=10000)
    do_csvr_graph(pl, constraints, data, 2, palette='fire')
    pl.show_grid()
    pl.show()
    pl.screenshot('figure_4.png')
