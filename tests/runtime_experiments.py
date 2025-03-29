"""
runtime_experiments
*******************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from convolutionalfixedsum.cfsvr import cfs, CFSError
import random
import time
import pickle
from tqdm import tqdm
from .util import UUniFastDiscard
from convolutionalfixedsum.cfsa import ivorfixedsum, IVoRFixedSum_Config
from drs import drs
import numpy as np

def drs_timer(n_tasks, seed, repeats=3):
    r = []
    random.seed(seed)
    uc = UUniFastDiscard(n_tasks, 1.5)
    np.random.seed(seed)
    for x in range(repeats):
        t1 = time.time()
        ret = drs(n_tasks, 1.0, uc)
        t2 = time.time()
    return r

def drs_time_experiments(outfn, n_exps, seed_offset, repeats=3):
    data = {}
    for x in tqdm(range(n_exps)):
        seed = seed_offset + x
        for n in tqdm(range(5, 51, 5)):
            if n not in data:
                data[n] = []
            data[n].extend(drs_timer(n, seed, repeats=repeats))
    with open(outfn, 'wb') as f: pickle.dump(data, f)


def cfsa_timer(n_tasks, seed, repeats=3):
    conf = IVoRFixedSum_Config(seed=seed)
    random.seed(seed)
    uc = UUniFastDiscard(n_tasks, 1.5)
    r = []
    for x in range(repeats):
        t1 = time.time()
        ret = ivorfixedsum(n_tasks, 1.0, uc=uc, config=conf)
        t2 = time.time()
        r.append(t2 - t1)
    return r


def cfsa_time_experiments(outfn, n_exps, seed_offset, repeats=3, timeout=100):
    data = {}
    for x in range(n_exps):
        seed = seed_offset + x
        # For CFSA, test with step size 1, because it's exponential
        # If doing step size of 5, execution time will jump by a factor of 128 - not great!
        for n in range(5, 51, 1):
            if n not in data: data[n] = []
            ts = cfsa_timer(n, seed, repeats=repeats)
            data[n].extend(ts)
            if any(t > timeout for t in ts):
                print(f'{n=} has time over {timeout}s, aborting.')
                break
    with open(outfn, 'wb') as f:
        pickle.dump(data, f)


def cfsvr_timer(n_tasks, seed, sample_size, repeats=3):
    random.seed(seed)
    uc = UUniFastDiscard(n_tasks, 1.5)
    r = []
    for x in range(repeats):
        t1 = time.time()
        try:
            ret = cfs(1.0, n_tasks, upper_constraints=uc, signal_size=sample_size, retries=1)
        except CFSError:
            pass
        t2 = time.time()
        r.append(t2 - t1)
    return r


def cfsvr_time_experiments(outfn, n_exps, seed_offset, sample_size, repeats=3):
    data = {}
    for x in tqdm(range(n_exps)):
        seed = seed_offset + x
        for n in tqdm(range(5, 51, 5)):
            if n not in data:
                data[n] = []
            ts = cfsvr_timer(n, seed, sample_size, repeats=repeats)
            data[n].extend(ts)
        with open(outfn, 'wb') as f:
            pickle.dump(data, f)
        

if __name__ == '__main__':
    for sample_size in [1000, 3000, 10000]:
        cfsvr_time_experiments(f'cfsvr-{sample_size}-time-data.pkl', 10, 0, sample_size)
    #cfsa_time_experiments('cfsa-time-data.pkl', 1, 0 )
    #drs_time_experiments('drs-time-data.pkl', 1, 0 )