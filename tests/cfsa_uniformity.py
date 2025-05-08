"""
cfsa_uniformity
***************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from convolutionalfixedsum.cfsa import CFSAConfig, ivorfixedsum
from .util import mk_bins_by_dimension, chisquare_stat, categorise_point, UUniFastDiscard
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

def cfsa_uniformity_test(uc, runs=10000, bins=10, seed=1, jumps=0):
    n_tasks= len(uc)
    conf = CFSAConfig(seed=seed, jumps=jumps)
    dist_by_dimension = {}
    try:
        bins_by_dimension = mk_bins_by_dimension(n_tasks, uc, 10, 'analytical')
    except:
        return None

    for x in range(n_tasks):
        dist_by_dimension[x] = [0] * bins

    t1 = time.time()
    for _ in tqdm(range(runs)):
        point = ivorfixedsum(n_tasks, 1, uc=uc, config=conf)
        categorise_point(n_tasks, bins_by_dimension, dist_by_dimension, uc, point)

    t2 = time.time()

    chisquare_results = chisquare_stat(dist_by_dimension)

    return dist_by_dimension, chisquare_results, t2 - t1, bins_by_dimension


def cfsa_mp_uniformity_test(n_tasks, runs, seed, uc_ratio=1.5):
    assert seed != -1
    t1 = time.time()
    try:
        random.seed(seed)
        uc = [x * uc_ratio for x in UUniFastDiscard(n_tasks, 1.0)]
        dist_by_dimension, chisquare_results, gen_duration, bins_by_dimension = \
            cfsa_uniformity_test(uc, runs=runs, seed=1+seed)  # 1+seed as xoroshiro256 seeds should be nonzero
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': uc_ratio, 'uc': uc, 'ret': True,
                'duration': t2 - t1, 'dist_by_dimension': dist_by_dimension, 'chisquare_results': chisquare_results,
                'gen_duration': gen_duration, 'bins_by_dimension': bins_by_dimension}
    except:
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': uc_ratio, 'ret': False}


def run_experiments_cfsa(
        outfn, min_tasks=3, max_tasks=10, n_exps=1000, runs_per_experiment=10000,
        seed_offset=0, uc_ratio=1.5
):
    tasks = []
    for n_tasks in tqdm(range(min_tasks, max_tasks + 1)):
        for seed in range(0, n_exps):
            tasks.append([n_tasks, seed + seed_offset])
    tasks.sort(reverse=True)
    futures = []
    results = []

    with ProcessPoolExecutor() as e:
        for n_tasks, seed in tasks:
            futures.append(e.submit(cfsa_mp_uniformity_test, n_tasks, runs_per_experiment, seed, uc_ratio))
        for f in tqdm(as_completed(futures), total=len(tasks)):
            results.append(f.result())
            if len(results) % 12 == 0:
                with open(outfn, 'wb') as f:
                    pickle.dump(results, f)

    with open(outfn, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    cfsa_uniformity_test([1] + [0.1] * 14, runs=10000, bins=10, seed=1, jumps=0)
    #run_experiments_cfsa('cfsa.pkl', 3, 15, 1000, 10000)
