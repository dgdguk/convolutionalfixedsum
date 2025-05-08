"""
cfsvf
*****

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license

Implementation of the Numeric ConvolutionalFixedSum method

Note: This version currently uses an old name, IVoRSFixedSum, which was later renamed to ConvolutionalFixedSum.
This will be fixed in a new version.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
import math
from .itp import itp, GOLDEN_RATIO
import random

class CFSError(Exception): ...

APPLY_FIX = True

if APPLY_FIX:
    def discretize_uc(uc, total, samples):
        assert total > 0
        ratio = min(uc / total, 1.0)
        signal_length = math.ceil(ratio * samples)
        if signal_length == 0:
            signal_length = 1
        r = signal_length / samples * total
        return r
else:
    def discretize_uc(uc, total, samples):
        # Slight error in this function: by doing a *floor*, some of the valid region is inaccessible.
        assert total > 0
        signal_length = math.floor(min(uc / total, 1.0) * samples)
        if signal_length == 0.0: signal_length = 1
        r = signal_length / samples * total
        return r

def calculate_signal_length(uc, total, samples):
    signal_length = int((min(uc / total, 1.0) * samples) + 0.5)
    if signal_length == 0.0:
        signal_length = 1
    return signal_length


def make_fft_signal(signal_length, padded_signal_length, val=1e-1):
    signal = np.zeros(padded_signal_length)
    signal[0:signal_length] = val
    return rfft(signal, padded_signal_length)

@dataclass
class CFSVR:
    """Solves a normalised CFS problem"""
    upper_constraints: List[float]
    total: float
    samples: int

    convolution_kernel: np.ndarray = field(init=False)
    total_volume: float = field(init=False)
    memo: Dict[float, float] = field(init=False)
    padded_signal_length: int = field(init=False)

    # ITP Configuration
    epsilon: float = 1e-6
    itp_k1: float = 1.0
    itp_k2: float = 0.98*(1+GOLDEN_RATIO)
    itp_n0: int = 1
    itp_enforce_max_iter: bool = True

    # Shrink threshold, factor
    normalization: bool = True



    def __post_init__(self):
        assert len(self.upper_constraints) > 1
        assert self.total >= 0
        self.memo = {}
        self.n_tasks = len(self.upper_constraints)
        signal_lengths = [calculate_signal_length(uc, self.total, self.samples) for uc in self.upper_constraints]
        self.padded_signal_length = next_fast_len(sum(signal_lengths), real=True) # .samples * self.n_tasks
        self.convolution_kernel = make_fft_signal(signal_lengths[1], self.padded_signal_length)
        for signal_length in signal_lengths[2:]:
            # TODO: Look at values inside convolution_kernel and see if I should rescale it
            # Think multiplying by a constant factor won't hurt, but obviously check
            if signal_length != 1:  # If signal length = 1, then this is equivalent to multiplying by 1. Skip.
                self.convolution_kernel *= make_fft_signal(signal_length, self.padded_signal_length)
            if self.normalization:
                mx_val = np.abs(self.convolution_kernel).max()
                self.convolution_kernel /= mx_val
        self.total_volume = self.volume_below(self.upper_constraints[0])
        if self.epsilon is None:
            self.epsilon = 1 / self.samples


    def volume_below(self, x):
        signal_length = calculate_signal_length(x, self.total, self.samples)
        if signal_length not in self.memo:
            if signal_length != 1:
                x_fft = make_fft_signal(signal_length, self.padded_signal_length)
                x_fft *= self.convolution_kernel
                inv = irfft(x_fft, self.padded_signal_length)
                res = inv[self.samples]
            else:
                res = irfft(self.convolution_kernel)[self.samples]
            self.memo[signal_length] = float(res)
        return self.memo[signal_length]

    def cdf(self, x):
        if x <= 0: return 0
        elif x >= self.upper_constraints[0]: return 1.0
        else: return float(self.volume_below(x) / self.total_volume)

    def inverse_cdf(self, x):
        """Uses ITP to calculate the inverse CDF"""
        return itp(self.cdf, 0, self.upper_constraints[0], x,
                   k1=self.itp_k1, k2=self.itp_k2, n0=self.itp_n0,
                   max_iter=self.itp_enforce_max_iter, epsilon=self.epsilon)


def __cfs(total, total_remaining, n, sorted_uc, signal_size):
    outc = []

    output = [-total]

    # Use CFSVR all the way down to 2 constraints
    for x in range(n - 2):
        if all(uc > total_remaining for uc in sorted_uc[x:]):
            # Upper constraints don't intersect the valid region, so we can use a UUniFast-like
            # approach
            rand = random.random()
            alloc = total_remaining * (1 - math.pow(1 - rand, 1.0 / (n - x - 1)))
            output.append(alloc)
            outc.append(sorted_uc[x])
        elif total_remaining > 0:
            # Discrete method; hitting upper edge of valid region has nonzero probability
            ds_uc = [discretize_uc(x, total_remaining, signal_size) for x in sorted_uc]
            cfsvr = CFSVR(ds_uc[x:], total_remaining, signal_size)
            r = random.random()
            alloc = cfsvr.inverse_cdf(r)
            if not APPLY_FIX:
                alloc = discretize_uc(alloc, total_remaining, signal_size)
            output.append(alloc)
            outc.append(ds_uc[x])
        else:  # If we hit an upper edge of valid region, no more utilisation is present
            output.append(0.0)
            outc.append(sorted_uc[x])
        # Update the total remaining
        total_remaining = max(-math.fsum(output), 0.0)

    # The last 2 constraints form a line segment, so it's easy to sample from.
    task_m2_lower_constraint = max(0.0, total_remaining - sorted_uc[
        -1])  # Implicit lower bound on penultimate task (don't leave too much for last task)
    task_m2_upper_constraint = min(sorted_uc[-2],
                                   total_remaining)  # Implicit upper bound on penultimate task (leave enough for last task)
    output.append(random.random() * (
                task_m2_upper_constraint - task_m2_lower_constraint) + task_m2_lower_constraint)  # Pick a uniform random point in the valid region
    output.append(max(-math.fsum(output), 0.0))  # Add in the final task.

    output.pop(0)
    return output


@dataclass
class CFSResult:
    total: float
    n: int
    lower_constraints: List[float]
    upper_constraints: List[float]
    signal_size: int
    rescale_output: bool
    rescale_triggered: bool
    retries: int
    retries_used: int
    output: List[float]


def cfsd(n, total, lower_constraints=None, upper_constraints=None, signal_size=10000, rescale_output=True, retries=1000):
    """
    Numerical approximation form of ConvolutionalFixedSum - Debug version. Same as cfs function, just
    returns some of the debug info.
    Args:
        n: Number of constraints.
        total: Total value to allocate. Defaults to 1.0. Probably should be positive.
        lower_constraints: Optional sequence of length n describing lower constraints. Defaults to all 0.
        upper_constraints: Optional sequence of length n describing upper constraints. Defaults to all total.
        signal_size: Signal size to use in approximation. Defaults to 10000, which is reasonable to 3DP with total=1.0
        rescale_output: Whether or not to rescale output - smoothes minor errors. Defautls to True.
        retries: Number of retries if a problem is encountered. Defaults to 1000.

    Returns:
        A CFSResult, with lots of debug info in it.
    """
    if lower_constraints is None:
        lower_constraints = [0] * n
    if upper_constraints is None:
        upper_constraints = [total] * n
    # Do some checks
    if sum(lower_constraints) >= total:
        raise ValueError(f"Sum of lower constraints ({sum(lower_constraints)}) >= total utilisation {total}")
    if sum(upper_constraints) <= total:
        raise ValueError(f"Sum of upper constraints ({sum(upper_constraints)}) <= total utilisation {total}")
    if any(x < 0 for x in lower_constraints):
        raise ValueError(f"Lower constraint < 0")
    if any(x < 0 for x in upper_constraints):
        raise ValueError(f"Upper constraint < 0")
    if len(lower_constraints) != n:
        raise ValueError(f"Lower constraints should be of length {n} ({len(lower_constraints)} supplied)")
    if len(upper_constraints) != n:
        raise ValueError(f"Upper constraints should be of length {n} ({len(upper_constraints)} supplied)")

    # Perform normalisation by setting all lower constraints to 1
    normalised_upper_constraints = [upper_constraints[x] - lower_constraints[x] for x in range(n)]
    total_remaining = modified_total = -math.fsum([-total] + lower_constraints)

    # Sort the upper constraints from low to high and make a map back
    sorted_uc, indexes = zip(*sorted(([uc, x] for x, uc in enumerate(normalised_upper_constraints)), reverse=True))
    for retries_used in range(retries):
        # __cfs's approximation is slightly above the valid region. So we try until we get a valid point.
        output  = __cfs(total, total_remaining, n, sorted_uc, signal_size)
        alloc_total = math.fsum(output)
        rescale_triggered = rescale_output and abs(alloc_total - modified_total) > 1e-4
        # Rescale minor infractions of the total back into the valid region - useful to counter floating point errors
        if rescale_triggered:
            output = [(x * modified_total) / alloc_total for x in output]
        # If we've found a valid point, we're now good,
        if all(0 <= output[x] <= sorted_uc[x] for x in range(n)):
            break
    else:
        raise CFSError("Was unable to draw from distribution successfully with {retries} attempts."
                       " Try narrowing the range of your constraints or increasing signal_size")

    # Restore the output to the order the user specified
    unsorted_output = [None] * n
    for x, i in enumerate(indexes):
        unsorted_output[i] = output[x]
    output = unsorted_output

    # Un-normalise the output:
    output = [output[x] + lower_constraints[x] for x in range(n)]

    if not all(lower_constraints[x] <= output[x] <= upper_constraints[x] for x in range(n)):
        raise CFSError("Un-normalising the problem caused violation of constraints, likely due to floating point error."
                       " Consider making the lower_constraints less significant.")

    return CFSResult(total, n, lower_constraints, upper_constraints, signal_size, rescale_output,
                         rescale_triggered, retries, retries_used, output
                        )


def cfs(n: int, total: float=1.0, lower_constraints: Optional[Sequence[float]]=None,
        upper_constraints: Optional[Sequence[float]]=None, signal_size: int=10000,
        rescale_output: bool=True, retries: int=1000):
    """
    Numerical approximation form of ConvolutionalFixedSum
    Args:
        n: Number of constraints.
        total: Total value to allocate. Defaults to 1.0. Probably should be positive.
        lower_constraints: Optional sequence of length n describing lower constraints. Defaults to all 0.
        upper_constraints: Optional sequence of length n describing upper constraints. Defaults to all total.
        signal_size: Signal size to use in approximation. Defaults to 10000, which is reasonable to 3DP with total=1.0
        rescale_output: Whether or not to rescale output - smoothes minor errors. Defautls to True.
        retries: Number of retries if a problem is encountered. Defaults to 1000.

    Returns:
        A vector of length n, sampling uniformly from the described area.
    """
    return cfsd(n, total, lower_constraints, upper_constraints, signal_size, rescale_output, retries).output


if __name__ == '__main__':
    random.seed(0)
    uc = [1.0]* 2 + [0.1] * 4
    r = cfsd(len(uc), 1.0, upper_constraints=uc, signal_size=100000)
    print(r.output)
