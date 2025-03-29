"""
figure_7
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from .cfsvr_uniformity import run_experiments_cfsvr
from .analyse_results import analyse_results

if __name__ == '__main__':
    # Parameters to this function can be varied
    run_experiments_cfsvr(
        'cfsvr.pkl',
        min_tasks=3,
        max_tasks=10,
        n_exps=1000,
        runs_per_experiment=10000,
        seed_offset=0,
        sample_signal_size=10000,
        uc_ratio=1.5
    )

    analyse_results('cfsvr.pkl', 'figure_7.pdf', 30)
