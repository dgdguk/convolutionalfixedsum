"""
figure_8
********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from .rfs_uniformity import run_experiments_rfs
from .analyse_results import analyse_results

if __name__ == '__main__':
    # Parameters to this function can be varied
    run_experiments_rfs(
        'rfs.pkl',
        min_tasks=3,
        max_tasks=10,
        # n_exps is deliberately lower here, due to RFS having less possibilities for constraints.
        # Instead, run_experiments_rfs internally generates constraints from 0.1 to 1.0 internally
        n_exps=10,
        runs_per_experiment=10000,
        seed_offset=0,
    )

    analyse_results('rfs.pkl', 'figure_8.pdf', 30)
