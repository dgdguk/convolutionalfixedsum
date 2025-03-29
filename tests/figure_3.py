"""
figure_3
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import pyvista as pv
from .viz import do_csvr_graph
from .uu_uniformity import uunifast_uniformity
from .util import mk_bins_by_dimension
import random

if __name__ == '__main__':
    random.seed(1)
    pl = pv.Plotter()
    uunifast_bins = mk_bins_by_dimension(3, [1.0]*3, 10, 'analytical')
    data = uunifast_uniformity(3, uunifast_bins, runs=10000)
    do_csvr_graph(pl, None, data, 2, palette='kgy')
    pl.show_grid()
    pl.show()
    pl.screenshot('figure_3.png')
