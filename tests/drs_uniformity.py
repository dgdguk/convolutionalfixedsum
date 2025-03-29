"""
drs_uniformity
**************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
import drs
import numpy as np

from .util import mk_bins_by_dimension, categorise_point, chisquare_stat, UUniFastDiscard

def drs_uniformity_test(uc, runs=10000, bins=10, tqdm_disable=False):
    n_tasks= len(uc)
    try:
        bins_by_dimension = mk_bins_by_dimension(n_tasks, uc, 10, 'analytical')
    except:
        return None

    dist_by_dimension = {}

    for x in range(n_tasks):
        dist_by_dimension[x] = [0] * bins

    generation_failures = 0
    unstable = 0
    t1 = time.time()

    for _ in tqdm(range(runs), disable=tqdm_disable):
        
        point = drs.drs(n_tasks, 1.0, upper_bounds=uc)
        categorise_point(n_tasks, bins_by_dimension, dist_by_dimension, uc, point)

    t2 = time.time()

    chisquare_results = chisquare_stat(dist_by_dimension)

    return dist_by_dimension, chisquare_results, t2-t1, generation_failures, unstable, bins_by_dimension


def mp_drs_uniformity_test(n_tasks, seed, uc_ratio, disable_inner_tqdm=False, runs=10000):
    t1 = time.time()
    try:
        random.seed(seed)
        np.random.seed(seed)

        uc = [x * uc_ratio for x in UUniFastDiscard(n_tasks, 1.0)]
        ret = drs_uniformity_test(uc, runs=runs, tqdm_disable=disable_inner_tqdm)
        dist_by_dimension, chisquare_results, gen_duration, generation_failures, unstable, bins_by_dimension = ret
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': uc_ratio, 'uc': uc, 'ret': ret,
                'duration': t2-t1,
                'dists_by_dimension': dist_by_dimension, 'chisquare_results': chisquare_results,
                'gen_duration': gen_duration, 'gen_failures': generation_failures, 'unstable': unstable, 'bins_by_dimension': bins_by_dimension
                }
    except:
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': uc_ratio, 'ret': None, 'duration': t2-t1}


def run_experiments_drs(outfn, min_tasks=3, max_tasks=10, n_exps=1000, runs_per_experiment=10000, seed_offset=0, uc_ratio=1.5):
    tasks = []
    for n_tasks in range(min_tasks,max_tasks+1):
        for seed in range(0, n_exps):
            tasks.append([n_tasks, seed+seed_offset])
    tasks.sort(reverse=True)
    futures = []
    results = []
    with ProcessPoolExecutor() as e:
        for n_tasks, seed in tasks:
            futures.append(e.submit(mp_drs_uniformity_test, n_tasks, seed, uc_ratio, True, runs=runs_per_experiment))
        for f in tqdm(as_completed(futures), total=len(tasks)):
            results.append(f.result())
            if len(results) % 12 == 0:
                with open(outfn, 'wb') as f:
                    pickle.dump(results, f)

    with open(outfn, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    run_experiments_drs('drs-uniformity.pkl', max_tasks=15)
