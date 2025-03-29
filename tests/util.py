"""
util
****

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import random
import numpy
import numpy as np
from scipy.stats import chisquare

from convolutionalfixedsum.cfsvr import CFSVR
from convolutionalfixedsum.cfsa import CFSADistribution


def chisquare_stat(dist_by_dimension):
    chisquare_results = {}
    for dim, dist in dist_by_dimension.items():
        chisquare_result = chisquare(dist)
        stat, pvalue = float(chisquare_result.statistic), float(chisquare_result.pvalue)
        chisquare_results[dim] = (stat, pvalue)
    return chisquare_results

def categorise_point(n_tasks, bins_by_dimension, dist_by_dimension, uc, point):
    assert (abs(1 - sum(point)) < 1e-1), sum(point)
    for x in range(n_tasks):
        b = 0
        assert point[x] <= uc[x], (point[x], uc[x])
        while point[x] > bins_by_dimension[x][b]:
            b += 1
        dist_by_dimension[x][b - 1] += 1


def UUniFastDiscard(n, u):
    while True:
        # Classic UUniFast algorithm:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)

        # If no task utilization exceeds 1:
        if all(ut <= 1 for ut in utilizations):
            return utilizations


def UUniFastDiscard_UC(n, u, uc):
    assert len(uc) == n
    while True:
        # Classic UUniFast algorithm:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)

        # If no task utilization exceeds 1:
        if all(ut <= uc for (ut, uc) in zip(utilizations, uc)):
            return utilizations

"""A taskset generator for experiments with real-time task sets

Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis. 
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation 
      and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are 
those of the authors and should not be interpreted as representing official 
policies, either expressed or implied, of Paul Emberson, Roger Stafford or 
Robert Davis.

Includes Python implementation of Roger Stafford's randfixedsum implementation
http://www.mathworks.com/matlabcentral/fileexchange/9700
Adapted specifically for the purpose of taskset generation with fixed
total utilisation value

Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have 
any questions regarding this software.
"""


def StaffordRandFixedSum(n, u, nsets):
    # deal with n=1 case
    if n == 1:
        return numpy.tile(numpy.array([u]), [nsets, 1])

    k = numpy.floor(u)
    s = u
    step = 1 if k < (k - n + 1) else -1
    s1 = s - numpy.arange(k, (k - n + 1) + step, step)
    step = 1 if (k + n) < (k - n + 1) else -1
    s2 = numpy.arange((k + n), (k + 1) + step, step) - s

    tiny = numpy.finfo(float).tiny
    huge = numpy.finfo(float).max

    w = numpy.zeros((n, n + 1))
    w[0, 1] = huge
    t = numpy.zeros((n - 1, n))

    for i in numpy.arange(2, (n + 1)):
        tmp1 = w[i - 2, numpy.arange(1, (i + 1))] * s1[numpy.arange(0, i)] / float(i)
        tmp2 = w[i - 2, numpy.arange(0, i)] * s2[numpy.arange((n - i), n)] / float(i)
        w[i - 1, numpy.arange(1, (i + 1))] = tmp1 + tmp2;
        tmp3 = w[i - 1, numpy.arange(1, (i + 1))] + tiny;
        tmp4 = numpy.array((s2[numpy.arange((n - i), n)] > s1[numpy.arange(0, i)]))
        t[i - 2, numpy.arange(0, i)] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (numpy.logical_not(tmp4))

    m = nsets
    x = numpy.zeros((n, m))
    rt = numpy.random.uniform(size=(n - 1, m))  # rand simplex type
    rs = numpy.random.uniform(size=(n - 1, m))  # rand position in simplex
    s = numpy.repeat(s, m);
    j = numpy.repeat(int(k + 1), m);
    sm = numpy.repeat(0, m);
    pr = numpy.repeat(1, m);

    for i in numpy.arange(n - 1, 0, -1):  # iterate through dimensions
        e = (rt[(n - i) - 1, ...] <= t[i - 1, j - 1])  # decide which direction to move in this dimension (1 or 0)
        sx = rs[(n - i) - 1, ...] ** (1 / float(i))  # next simplex coord
        sm = sm + (1 - sx) * pr * s / float(i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    # iterated in fixed dimension order but needs to be randomised
    # permute x row order within each column
    for i in range(0, m):
        x[..., i] = x[numpy.random.permutation(n), i]

    return numpy.transpose(x);


def rfs(n, l, nsets):
    """RandFixedSum, but specifying a limit instead of max utilisation"""
    u = 1 / l
    r = StaffordRandFixedSum(n, u, nsets)
    return r / u


def mk_bins_by_dimension(n_tasks, uc, bins, slice_calc):
    bins_by_dimension = {}
    uc_rot = uc
    for x in range(n_tasks):
        if slice_calc == 'analytical':
            dist = CFSADistribution(n_tasks, 1.00, None, uc_rot)
            bins_by_dimension[x] = [dist.icdf(y) for y in np.arange(0.0, 1.01, 1.0 / bins)]
        elif isinstance(slice_calc, int):
            dist = CFSVR(uc, 1.00, slice_calc)
            bins_by_dimension[x] = [float(dist.inverse_cdf(y)) for y in np.arange(0.0, 1.01, 1.0 / bins)]
        else:
            raise ValueError(f'slice calc should be "analytical" or an integer, got {slice_calc}')
        uc_rot.append(uc_rot.pop(0))
    return bins_by_dimension