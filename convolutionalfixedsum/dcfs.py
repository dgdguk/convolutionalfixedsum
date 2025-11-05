"""
dcfs
****

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license

Binding to the C implementation of the Analytical CFSA method

Note: This version currently uses an old name, IVoRSFixedSum, which was later renamed to ConvolutionalFixedSum.
This will be fixed in a new version.
"""

from convolutionalfixedsum import cfsa, cfsn, CFSAConfig
from typing import Optional, Sequence
import numpy as np
import itertools
import random

class DCFSError(Exception): ...


def enumerate_points(n: int,
         total: float=1.0, lower_constraints: Optional[Sequence[float]]=None,
         upper_constraints: Optional[Sequence[float]]=None,
         lattice_spacing: Optional[Sequence[float]]=None,
         lattice_origin: Optional[Sequence[float]]=None,
         tolerance: Optional[float]=None,
         rounding: int=4,
         maxhypersize: int=60466176):
    """Enumerate all points describe by the given lattice, subject to the given constraints"""

    if lattice_spacing is None or lattice_origin is None or tolerance is None:
        raise ValueError('Missing Argument')

    if lower_constraints is None:
        lower_constraints = [0.0] * n

    if upper_constraints is None:
        upper_constraints = [total + tolerance] * n

    lattice_points_by_dim = []
    hypercube_points = 1
    for lc, uc, spacing, origin in zip(lower_constraints, upper_constraints, lattice_spacing, lattice_origin):
        points = []
        point = lc - ((lc + origin) % spacing)
        if point < lc: point += spacing
        while point <= uc:
            points.append(point)
            point += spacing
        lattice_points_by_dim.append(points)
        hypercube_points *= len(points)

    if hypercube_points > maxhypersize:
        raise ValueError(f"Too many lattice points in hypercube {hypercube_points} {maxhypersize}")

    res = [tuple(round(y, rounding) for y in x) for x in itertools.product(*lattice_points_by_dim)
           if total - tolerance <= sum(x) <= total + tolerance]
    if len(set(res)) != len(res):
        raise ValueError("Need more precise rounding")
    return res

def dcfs(
        n: int,
        total: float=1.0,
        lower_constraints: Optional[Sequence[float]]=None,
        upper_constraints: Optional[Sequence[float]]=None,
        lattice_spacing: Optional[Sequence[float]]=None,
        lattice_origin: Optional[Sequence[float]]=None,
        tolerance: Optional[float]=None,
        max_retries: int=10000,
        expansion_mode: str='correct',
        tolerance_expansion_mode: str='correct',
        cfs_mode='analytical',
        cfsa_config=None,
        cfsn_signal_size=10000
):
    """Sample a point from the given lattice, subject to the given constraints"""

    if cfs_mode == 'analytical':
        cfs_func = lambda n, t, lc, uc: cfsa(n, t, lc, uc, config=cfsa_config)
    elif cfs_mode == 'numeric':
        cfs_func = lambda n, t, lc, uc: cfsn(n, t, lc, uc, signal_size=cfsn_signal_size)
    else:
        raise ValueError('Incorrect CFS mode')

    if lattice_spacing is None or lattice_origin is None or tolerance is None:
        raise ValueError('Missing Argument')

    if lower_constraints is None:
        lower_constraints = [0.0] * n

    if upper_constraints is None:
        upper_constraints = [total + tolerance] * n

    orig_lc_array = np.asarray(lower_constraints)
    orig_uc_array = np.asarray(upper_constraints)


    ls_array = np.asarray(lattice_spacing)
    lo_array = np.asarray(lattice_origin) % ls_array

    lc_array = orig_lc_array - lo_array
    uc_array = orig_uc_array - lo_array

    if tolerance_expansion_mode == 'alt':
        expanded_tolerance = np.max(ls_array) + tolerance
    elif tolerance_expansion_mode == 'correct':
        expanded_tolerance = (np.sum(ls_array) / 2) + tolerance
    elif tolerance_expansion_mode == 'none':
        expanded_tolerance = tolerance
    else:
        raise ValueError()
    expanded_lower_constraints = np.ndarray(n + 1)
    expanded_lower_constraints[0:n] = lc_array
    if expansion_mode == 'max':
        expanded_lower_constraints[0:n] -= ls_array
    elif expansion_mode == 'correct':
        expanded_lower_constraints[0:n] -= expanded_lower_constraints[0:n] % ls_array
        expanded_lower_constraints[0:n] -= ls_array / 2
    elif expansion_mode == 'none':
        pass
    else:
        raise ValueError()
    expanded_lower_constraints[n] = -expanded_tolerance

    expanded_upper_constraints = np.ndarray(n + 1)
    expanded_upper_constraints[0:n] = uc_array
    if expansion_mode == 'max':
        expanded_upper_constraints[0:n] += ls_array
    elif expansion_mode == 'correct':
        rem = expanded_upper_constraints[0:n] % ls_array
        if np.any(rem):
            expanded_upper_constraints[0:n] += ls_array - rem
        expanded_upper_constraints[0:n] += ls_array / 2
    elif expansion_mode == 'none':
        pass
    else:
        raise ValueError()
    expanded_upper_constraints[n] = expanded_tolerance

    count = 0
    while count <= max_retries:
        p_prime = list(cfs_func(n + 1, total, expanded_lower_constraints, expanded_upper_constraints))[:-1]
        p = p_prime + (ls_array / 2.0)
        p = p - (p % ls_array) + lo_array
        if np.all(orig_lc_array - 1e-4 <= p) and np.all(p <= orig_uc_array + 1e-4) and np.abs(np.sum(p) - total) <= (tolerance + 1e-8):
            return p, count
        count += 1

    raise DCFSError('Max Retries exceeded')


def dcfsa(
        n: int,
        total: float=1.0,
        lower_constraints: Optional[Sequence[float]]=None,
        upper_constraints: Optional[Sequence[float]]=None,
        lattice_spacing: Optional[Sequence[float]]=None,
        lattice_origin: Optional[Sequence[float]]=None,
        tolerance: Optional[float]=None,
        max_retries: int=10000,
        config=None,
):
    """Generate a point using DCFS, using the analytical mode of CFS for sampling"""
    return dcfs(n, total, lower_constraints, upper_constraints, lattice_spacing, lattice_origin,
                tolerance, max_retries, cfs_mode='analytical', cfsa_config=config)[0]

def dcfsn(
        n: int,
        total: float=1.0,
        lower_constraints: Optional[Sequence[float]]=None,
        upper_constraints: Optional[Sequence[float]]=None,
        lattice_spacing: Optional[Sequence[float]]=None,
        lattice_origin: Optional[Sequence[float]]=None,
        tolerance: Optional[float]=None,
        max_retries: int=10000,
        signal_size=10000
):
    """Generate a point using DCFS, using the numeric mode of CFS for sampling"""
    return dcfs(n, total, lower_constraints, upper_constraints, lattice_spacing, lattice_origin,
                tolerance, max_retries, cfs_mode='numeric', cfsn_signal_size=signal_size)[0]


def enumeration_sampling(n: int, k: int,
         total: float=1.0, lower_constraints: Optional[Sequence[float]]=None,
         upper_constraints: Optional[Sequence[float]]=None,
         lattice_spacing: Optional[Sequence[float]]=None,
         lattice_origin: Optional[Sequence[float]]=None,
         tolerance: Optional[float]=None,
         rounding: int=4,
         maxhypersize: int=60466176):
    """Generate k points sampled using enumeration sampling"""
    points_list = enumerate_points(n, total, lower_constraints, upper_constraints, lattice_spacing, lattice_origin,
                                   tolerance, rounding, maxhypersize)
    return random.choices(points_list, k=k)
