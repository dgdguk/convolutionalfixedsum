"""
itp
***

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license

A Python implementation of ITP [1] root-finding, with some minor modifications
that relax some of the constraints of the pseudocode implementation in [1].

ITP is an algorithm that find a root of a function within a given interval.
It has the superlinear convergence speed of the secant root-finding method,
whilst not suffering from non-convergence issues. Instead, in the worst-case,
ITP performs as well as a binary search / bisection method. The average
performance of ITP is strictly better than binary search / bisection methods.

In their paper, Oliveira and Takahashi [1] showed that ITP performs better
than other widely used general purpose root-finding methods, showing a
substantial performance increase over Brent's Method, Ridders' Method, and
the Illinois Algorithm. While specialised algorithms, such as Newton's method,
may exhibit higher performance, they have restrictions or non-convergence
issues that ITP does not have.

[1] I. F. D. Oliveira and R. H. C. Takahashi. 2020.
An Enhancement of the Bisection Method Average Performance Preserving Minmax
Optimality. ACM Trans. Math. Softw. 47, 1, Article 5 (March 2021), 24 pages.
https://doi.org/10.1145/3423597
"""
from __future__ import annotations

import math
from typing import Callable, Optional
import warnings

GOLDEN_RATIO = 0.5 * (1 + math.sqrt(5))


def itp_max_iter(a: float, b: float, epsilon: float=1e-10, n0: int=1) -> float:
    """Returns the theoretical max number of iterations required for ITP method
    Arguments:
        a,b: interval on which to calculate
        epsilon: Required level of accuracy, default 1e-10
        n0: ITP hyperparameter.
    """
    return int(math.ceil(math.log2((abs(b - a))/(2*epsilon))) + n0)


def itp(func: Callable[[float], float], a: float, b: float, c: float=0.0,
        epsilon: float=1e-10, k1: Optional[float]=0.2, 
        k2: float=0.99*(1+GOLDEN_RATIO), n0: int=1,
        max_iter: Optional[bool, int]=None) -> float:
    """ITP (Interpolate, Truncate, Project) root finding method.
    Arguments:
        func: The function to search over
        a, b: The interval in which a f(x) = c lies
        c: Find intersections with line x=c; defaults to 0 (roots)
        epsilon: Required level of accuracy, default 1e-10
        k1, k2, n0: Hyperparameters of ITP method. Default values are derived
        from [1], and should be a) sane and b) lead to good performance.
        max_iter: If true, enforce the theoretical max number of iterations.
                  If a number, enforce this number of iterations.
    """
    # Comments referring to Olivera and Takahashi refer to Algorithm 1, in [1]
    if a == b: raise ValueError('a and b must not be equal')
    # Olivera and Takahashi assume a < b, so swap them if this is not true.
    if a > b:
        a, b = b, a
    y_a = func(a) - c
    y_b = func(b) - c
    if abs(y_a) <= epsilon:
        return a
    elif abs(y_b) <= epsilon:
        return b

    # Pseudocode in Olivera and Takahashi assumes y_a < 0 < y_b.
    # To ensure this, we'll take the sign of y_b and multiply everything we obtain from
    # func by it, creating a derived function that obeys the inequality.
    direction = math.copysign(1, y_b)
    y_a *= direction
    y_b *= direction

    if max_iter is True:
        max_iter = itp_max_iter(a, b, epsilon, n0)

    assert y_a < 0 and y_b > 0, f"func({a})={y_a} and func({b})={y_b} must be on opposite sides of zero"
    
    if k1 is None:
        k1 = 0.2 / (b - a)
    # Hyperparameter check
    assert k1 > 0, "Hyperparamter k1 must be positive"
    assert 1 <= k2 < 1 + GOLDEN_RATIO, "Hyperparameter k2 must be between 1 and 1 + 0.5*(1+math.sqrt(5))"
    assert n0 >= 0, "Hyperparameter n0 must be >= 0"
    if n0 == 0: warnings.warn('Setting n0 == 0 has the potential to cause numerical instability, '
                              'and this implementation of ITP does not check against this')

    n_half = math.ceil(math.log2((b - a)/(2*epsilon)))
    n_max = n_half + n0
    k = 0

    while (b - a > 2 * epsilon):
        # Interpolate
        x_f = (y_b*a - y_a*b) / (y_b - y_a)
        # Truncate
        x_half = (a + b) / 2
        sigma = math.copysign(1, x_half - x_f)
        delta = k1 * ((b - a) ** k2)
        if delta <= abs(x_half - x_f):
            x_t = x_f + math.copysign(delta, sigma)
        else:
            x_t = x_half
        # Projection, equation(15)
        r = epsilon * (2**(n_max - k)) - (b - a) / 2
        if abs(x_t - x_half) <= r:
            x_itp = x_t
        else:
            x_itp = x_half - math.copysign(r, sigma)
        # Update interval
        y_itp = (func(x_itp) - c) * direction
        if y_itp > 0:
            b = x_itp
            y_b = y_itp
        elif y_itp < 0:
            a = x_itp
            y_a = y_itp
        else:
            a = b = x_itp
        if max_iter is not None and k > (max_iter):
            raise Exception(f'Non-convergence detected (precision {abs(a-b)}); is your function precise enough? (max_iter={max_iter})')
        k += 1
    result = (a + b) / 2
    return result


if __name__ == '__main__':
    print(itp_max_iter(0, 1))
    def f(x): return 3*x**2 -2*x + -4
    def f_offset_by_2(x): return f(x) + 2
    roots = [1.5352, -0.8685]  # Approximate to 4 decimal places
    print(roots)
    print(f(roots[0]), f(roots[1]))
    print(itp(f, 1.0, 2.0), itp(f, 0, -1))
    print(f_offset_by_2(roots[0]), f_offset_by_2(roots[1]))
    print(itp(f_offset_by_2, 1.0, 2.0, 2.0))