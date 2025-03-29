"""
uu_uniformity
*************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
from .util import UUniFastDiscard, mk_bins_by_dimension, categorise_point, chisquare_stat


def uunifast_uniformity(n_tasks, bins_by_dimension, runs=1000, bins=10, tqdm_disable=False):
    assert bins_by_dimension is not None
    dist_by_dimension = {}
    uc = [1.0] * n_tasks

    for x in range(n_tasks):
        dist_by_dimension[x] = [0] * bins

    generation_failures = 0
    unstable = 0
    t1 = time.time()

    points = []
    for _ in tqdm(range(runs), disable=tqdm_disable):
        points.append(UUniFastDiscard(n_tasks, 1))

    for point in points:
        categorise_point(n_tasks, bins_by_dimension, dist_by_dimension, uc, point)

    t2 = time.time()

    chisquare_results = chisquare_stat(dist_by_dimension)

    return dist_by_dimension, chisquare_results, t2-t1, generation_failures, unstable, bins_by_dimension


def uunifast_mp_uniformity(n_tasks, seed, bins_by_dimension, runs, disable_inner_tqdm=False):
    random.seed(seed)
    t1 = time.time()
    try:
        ret = uunifast_uniformity(n_tasks, bins_by_dimension, runs=runs, tqdm_disable=disable_inner_tqdm)
        dist_by_dimension, chisquare_results, gen_duration, generation_failures, unstable, bins_by_dimension = ret
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'ret': True,
                'duration': t2-t1,
                'dists_by_dimension': dist_by_dimension, 'chisquare_results': chisquare_results,
                'gen_duration': gen_duration, 'gen_failures': generation_failures, 'unstable': unstable, 'bins_by_dimension': bins_by_dimension
                }
    except:
        # Emergency handle any error case, get enough data that the error can be reproduced
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'ret': None, 'duration': t2-t1}


def run_experiments_uunifast(outfn, min_tasks=3, max_tasks=10, n_exps=1000, runs_per_experiment=10000, seed_offset=0):
    tasks = []
    b = {}
    for n_tasks in tqdm(range(min_tasks, max_tasks+1)):
        b[n_tasks] = mk_bins_by_dimension(n_tasks, [1.0]*n_tasks, 10, 'analytical')
        for seed in range(0, n_exps):
            tasks.append([n_tasks, seed+seed_offset])
    tasks.sort(reverse=True)
    futures = []
    results = []
    
    with ProcessPoolExecutor() as e:
        for n_tasks, seed in tasks:
            futures.append(e.submit(uunifast_mp_uniformity, n_tasks, seed, b[n_tasks], runs_per_experiment, True))
        for f in tqdm(as_completed(futures), total=len(tasks)):
            results.append(f.result())
            if len(results) % 12 == 0:
                with open(outfn, 'wb') as f:
                    pickle.dump(results, f)

    with open(outfn, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    run_experiments_uunifast('uunifast.pkl', max_tasks=15)
