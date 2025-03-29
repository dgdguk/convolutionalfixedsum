"""
cfsvr_uniformity
****************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from convolutionalfixedsum.cfsvr import cfsd
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
import os

from .util import UUniFastDiscard, chisquare_stat, categorise_point, mk_bins_by_dimension


def cfsvr_uniformity_test(uc, runs=1000, bins=10, tqdm_disable=False, sample_signal_size=10000):
    n_tasks= len(uc)
    dist_by_dimension = {}
    dist_by_dimension_stableonly = {}

    try:
        bins_by_dimension = mk_bins_by_dimension(n_tasks, uc, 10, 'analytical')
    except:
        return None
    for x in range(n_tasks):
        dist_by_dimension[x] = [0] * bins
        dist_by_dimension_stableonly[x] = [0] * bins

    generation_failures = 0
    unstable = 0
    t1 = time.time()

    for _ in tqdm(range(runs), disable=tqdm_disable):
        while True:
            try:
                result = cfsd(1, n_tasks, upper_constraints=uc, signal_size=sample_signal_size)
                point = result.output
                if result.rescale_triggered: unstable += 1
                break
            except:
                generation_failures += 1
                if generation_failures >= (runs * 2): return None

        categorise_point(n_tasks, bins_by_dimension, dist_by_dimension, uc, point)
        if not result.rescale_triggered: categorise_point(n_tasks, bins_by_dimension, dist_by_dimension_stableonly, uc, point)

    t2 = time.time()

    chisquare_results = chisquare_stat(dist_by_dimension)
    chisquare_results_stableonly = chisquare_stat(dist_by_dimension_stableonly)

    return dist_by_dimension, chisquare_results, dist_by_dimension_stableonly, chisquare_results_stableonly, t2-t1, generation_failures, unstable, bins_by_dimension


def mp_cfsvr_uniformity_test(n_tasks, seed, uc_ratio, disable_inner_tqdm=False, runs=1000, sample_signal_size=10000):
    try:
        random.seed(seed)
        t1 = time.time()
        uc = [x * uc_ratio for x in UUniFastDiscard(n_tasks, 1.0)]
        ret = cfsvr_uniformity_test(uc, runs=runs, tqdm_disable=disable_inner_tqdm, sample_signal_size=sample_signal_size)
        dist_by_dimension, chisquare_results, dist_by_dimension_stableonly, chisquare_results_stableonly, gen_duration, generation_failures, unstable, bins_by_dimension = ret
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': uc_ratio, 'uc': uc, 'ret': ret,
                'duration': t2-t1, 'sample-signal': sample_signal_size,
                'dists_by_dimension': dist_by_dimension, 'chisquare_results': chisquare_results,
                'dists_by_dimension_stable': dist_by_dimension_stableonly, 'chisquare_results_stable': chisquare_results_stableonly,
                'gen_duration': gen_duration, 'gen_failures': generation_failures, 'unstable': unstable, 'bins_by_dimension': bins_by_dimension
                }
    except:
        t2 = time.time()
        return {'n_tasks': n_tasks, 'seed': seed, 'uc_ratio': uc_ratio, 'uc': uc, 'ret': None, 'duration': t2-t1,
                'sample-signal': sample_signal_size}


def run_experiments_cfsvr(outfn, min_tasks=3, max_tasks=10, n_exps=1000, runs_per_experiment=10000,
                          seed_offset=0, sample_signal_size=10000, uc_ratio=1.5):
    tasks = []
    for n_tasks in range(min_tasks, max_tasks+1):
        for seed in range(0, n_exps):
            tasks.append([n_tasks, seed + seed_offset])
    #tasks.sort(reverse=True)
    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as e:
        for n_tasks, seed in tasks:
            futures.append(e.submit(mp_cfsvr_uniformity_test, n_tasks, seed, uc_ratio, True, runs_per_experiment, sample_signal_size))
        for f in tqdm(as_completed(futures), total=len(tasks)):
            results.append(f.result())
            if len(results) % 12 == 0:
                with open(outfn, 'wb') as f:
                    pickle.dump(results, f)

    with open(outfn, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    random.seed(0)
    l = 1.0
    s = 1e-2
    vs = 1e-2
    x = cfsvr_uniformity_test([l]*2 + [s]*11, sample_signal_size=1000)
    print(x[0])
    print(x[1])
    #run_experiments_cfsvr('cfsvr-a-1000.pkl', sample_signal_size=1000)
    #run_experiments_cfsvr('cfsvr-a-10000.pkl', sample_signal_size=10000)


