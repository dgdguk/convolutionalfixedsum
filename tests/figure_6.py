"""
figure_6
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from .cfsa_uniformity import run_experiments_cfsa
from .analyse_results import analyse_results

if __name__ == '__main__':
    # Parameters to this function can be varied
    run_experiments_cfsa(
        'cfsa.pkl',
        min_tasks=3,
        max_tasks=10,
        n_exps=1000,
        runs_per_experiment=10000,
        seed_offset=0,
        uc_ratio=1.5
    )

    analyse_results('cfsa.pkl', 'figure_6.pdf', 30)
