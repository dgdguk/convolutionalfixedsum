"""
rfs_uniformity
**************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
from .util import rfs, mk_bins_by_dimension, categorise_point, chisquare_stat
import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def rfs_uniformity_test(n_tasks, limit, runs=1000, bins=10, tqdm_disable=False):
    dist_by_dimension = {}
    uc = [limit] * n_tasks

    try:
        bins_by_dimension = mk_bins_by_dimension(n_tasks, uc, 10, 'analytical')
        print(bins_by_dimension)
    except:
        return None
    for x in range(n_tasks):
        dist_by_dimension[x] = [0] * bins

    generation_failures = 0
    unstable = 0
    t1 = time.time()

    points = rfs(n_tasks, limit, runs)

    for point in points:
        categorise_point(n_tasks, bins_by_dimension, dist_by_dimension, uc, point)

    t2 = time.time()

    chisquare_results = chisquare_stat(dist_by_dimension)

    return dist_by_dimension, chisquare_results, t2-t1, generation_failures, unstable, bins_by_dimension


def rfs_mp_uniformity_test(n_tasks, seed, limit, disable_inner_tqdm=False, runs=10000):
    random.seed(seed)
    t1 = time.time()
    np.random.seed(seed)
    try:
        ret = rfs_uniformity_test(n_tasks, limit, runs=runs, tqdm_disable=disable_inner_tqdm)
        dist_by_dimension, chisquare_results, gen_duration, generation_failures, unstable, bins_by_dimension = ret
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'limit': limit, 'ret': True,
                'duration': t2-t1,
                'dists_by_dimension': dist_by_dimension, 'chisquare_results': chisquare_results,
                'gen_duration': gen_duration, 'gen_failures': generation_failures, 'unstable': unstable, 'bins_by_dimension': bins_by_dimension
                }
    except:
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': limit, 'ret': None, 'duration': t2-t1}

def run_experiments_rfs(outfn, min_tasks=3, max_tasks=10, n_exps=1000, runs_per_experiment=10000, seed_offset=0):
    tasks = []
    for n_tasks in range(min_tasks, max_tasks+1):
        random.seed(n_tasks+seed_offset)
        for seed in range(0, n_exps):
            i_limit = random.random()
            while i_limit < 1e-2: i_limit = random.random()
            m_limit = i_limit * (1 - 1/n_tasks) + (1 / n_tasks)
            tasks.append([n_tasks, seed+seed_offset, m_limit])
        #for limit in np.arange(0.0, 1.01, 0.1):
        #    m_limit = limit * (1 - 1/n_tasks) + (1 / n_tasks)
        #    print(m_limit)
        #    if limit * n_tasks > 1.01:
        #        for seed in range(0, n_exps):
        #            ... #tasks.append([n_tasks, seed+seed_offset, limit])
    tasks.sort(reverse=True)
    futures = []
    results = []
    with ProcessPoolExecutor() as e:
        for n_tasks, seed, limit in tasks:
            futures.append(e.submit(rfs_mp_uniformity_test, n_tasks, seed, limit, True, runs_per_experiment))
        for f in tqdm(as_completed(futures), total=len(tasks)):
            results.append(f.result())
            if len(results) % 12 == 0:
                with open(outfn, 'wb') as f:
                    pickle.dump(results, f)

    with open(outfn, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    n = 9
    r = rfs_uniformity_test(n, 1e-3 * (1 - 1/n) + (1 / n))
    print(r[0])
    #run_experiments_rfs('rfs-uniformity-a.pkl', max_tasks=15)
