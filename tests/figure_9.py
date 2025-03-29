"""
figure_9
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from .drs_uniformity import run_experiments_drs
from .analyse_results import analyse_results

if __name__ == '__main__':
    # Parameters to this function can be varied
    run_experiments_drs(
        'drs.pkl',
        min_tasks=3,
        max_tasks=10,
        n_exps=1000,
        runs_per_experiment=10000,
        seed_offset=0,
        uc_ratio=1.5
    )

    analyse_results('drs.pkl', 'figure_9.pdf', 150)
