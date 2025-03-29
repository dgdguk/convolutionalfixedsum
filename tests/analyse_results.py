"""
analyse_results
***************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import pickle
import sys

import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest
import numpy as np

#with open('cfsvr-c-10000.pkl', 'rb') as f:
#    d = pickle.load(f)
#fd = [x for x in d if x['n_tasks'] != 10]
#with open('cfsvr-c-f-10000.pkl', 'wb') as f:
#    pickle.dump(fd, f)
#assert False

files = {
  'CFSA': (['cfsa.pkl', 'cfsa-b.pkl'], 'cumulative-graph-cfsa.pdf', 30),
  'CFSD': ('cfsvr-a-10000.pkl', 'cumulative-graph-convolutionalfixedsum.pdf', 30),
  'UU': ('uunifast.pkl', 'cumulative-graph-uu.pdf', 30),
  'RFS': ('rfs-uniformity.pkl', 'cumulative-graph-rfs.pdf', 30),
  'DRS': ('drs-uniformity.pkl', 'cumulative-graph-drs.pdf', 150),
  'CFSD2': (['cfsvr-b-10000.pkl', 'cfsvr-c-f-10000.pkl'], 'cumulative-graph-cfs2.pdf', 30),
  'RFS2': ('rfs-uniformity-a.pkl', 'cumulative-graph-rfs2.pdf', 30),
}

golden = (1 + 5**0.5)/2


def analyse_results(in_fn, graph_fn, filter_limit=150):
    if isinstance(in_fn, list):
        data = []
        for fn in in_fn:
            with open(fn, 'rb') as f:
                d = pickle.load(f)
                data.extend(d)
    elif isinstance(in_fn, str):
        with open(in_fn, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError('Unknown what in_fn is')

    absolute_fails = 0
    failed_tests = 0
    total_tests = 0
    stats = []
    unf = []
    for d in data:
        if 'chisquare_results' in d:
            chisquares = d['chisquare_results']
            for k, v in chisquares.items():
                total_tests += 1
                if v[1] < 0.05:
                    failed_tests += 1
                    #print(d)
                if v[0] <= filter_limit:
                    stats.append(v[0])
                unf.append(v[0])
        else:
            absolute_fails += d['n_tasks']

    n_bins = 50

    dist1 = stats

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    fig.set_layout_engine('compressed')
    #fig.tight_layout(h_pad=0.0, w_pad=1.08)
    #axs.set_aspect(golden)

    axs.hist(dist1, bins=n_bins, density=True, label='Observed Distribution')

    x = np.arange(0, 30, 0.001)
    axs.plot(x, chi2.pdf(x, df=9), label='χ$^2$ (9 DoF)')
    axs.legend()
    plt.savefig(graph_fn)
    factor = 1.0
    stats = [x*factor for x in stats]
    unf = [x*factor for x in unf]
    kstest_result = kstest(stats, lambda x: chi2.cdf(x, df=9))
    kstest_result_unf = kstest(unf, lambda x: chi2.cdf(x, df=9))
    #kstest_lt_result = kstest(stats, lambda x: chi2.cdf(x, df=9), alternative='less')
    #kstest_lt_result_unf = kstest(unf, lambda x: chi2.cdf(x, df=9), alternative='less')

    print("Reported stats: ")
    print(f'Total number of considered tests: {len(stats)}')
    print(f'Maximum chi2 Statistic observed (use for setting graph limit): {max(stats)}')
    print(f'Number of chi2 tests exceeding p-value of 0.05: {failed_tests}')
    print('Note for evaluators: chi2 tests will exceed this p-value by random chance')
    print(f'KS test result for full data set, alternative equals: {kstest_result_unf}')
    print(f'KS test result for graph data set, alternative equals: {kstest_result}')
    #print(f'KS test result for full data set, alternative less: {kstest_lt_result_unf}')
    #print(f'KS test result for graph data set, alternative less: {kstest_lt_result}')
    print('Note for evaluators: KS test is checked for both the full data set and graph to alert user if'
          ' the data set used for the graph may be misleading.')
    print(f'Number of tests that failed to run due to error checking in algorithm: {absolute_fails}')
    print(total_tests, failed_tests, (failed_tests / total_tests)*100, len(stats))

if __name__ == '__main__':
    analyse_results(*files['RFS2'])
    #for key in files:
    #    print(key)
    #    analyse_results(*files[key])
