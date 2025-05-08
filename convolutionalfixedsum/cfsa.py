"""
cfsa
****

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license

Binding to the C implementation of the Analytical CFSA method

Note: This version currently uses an old name, IVoRSFixedSum, which was later renamed to ConvolutionalFixedSum.
This will be fixed in a new version.
"""
from _cfsa import ffi, lib
import numpy as np
from dataclasses import dataclass, field 
from typing import Optional, Iterable, Sequence

class CFSAError(Exception): ...

@dataclass
class CFSAConfig:
    """
    Configuration for analytical CFS

    seed / jumps: Configuration for the RNG.
    epsilon: Accuracy with which to conduct root finding to
    itp_params: A dictionary for ITP parameters - currently unused, reserved for future tuning
    relative_epsilon: If the epsilon parameter should be scaled for current utilisation
    minimum_epsilon: If using relative epsilon, a minimum value for epsilon (as FP error will happen)
    """
    seed: Optional[int] = None
    jumps: Optional[int] = None
    epsilon: Optional[float] = 1e-10
    itp_params: Optional[dict] = None
    relative_epsilon: bool = True 
    minimum_epsilon: float = 1e-15

    def __post_init__(self):
        if self.itp_params: raise NotImplementedError()
        self.ivorfs_config = ffi.new("IVoRFS_Config*")
        self.ivorfs_config.itp_config = ffi.NULL
        self.ivorfs_config.epsilon = self.epsilon
        if self.seed or self.jumps:
            self.prng = ffi.new("PluggableRNG*")
            lib.pluggable_rand_xoroshiro256_rng_init(self.prng)
            self.ivorfs_config.rf = self.prng
            if self.seed:
                lib.pluggable_rand_seed(self.prng, self.seed)
            if self.jumps:
                for _ in range(self.jumps):
                    lib.pluggable_rand_jump(self.prng)
        else:
            self.prng = None

    def __del__(self):
        if self.prng is not None:
            lib.pluggable_rand_xoroshiro256_rng_uninit(self.prng)


def ivorfixedsum_default_seed(seed):
    lib.pluggable_rand_seed(ffi.NULL, seed)

def ivorfixedsum(n_constraints, total, lc=None, uc=None, config=None):
    if lc is not None:
        assert len(lc) == n_constraints
        lc = np.asarray(lc, dtype=np.float64)
        p_lc = ffi.cast("double *", lc.ctypes.data)
    else:
        p_lc = ffi.NULL
    if uc is not None:
        assert len(uc) == n_constraints
        uc = np.asarray(uc, dtype=np.float64)
        p_uc = ffi.cast("double *", uc.ctypes.data)
    else:
        p_uc = ffi.NULL
    if config is not None:
        p_config = config.ivorfs_config
    else:
        p_config = ffi.NULL
    result = np.zeros(n_constraints, dtype=np.float64)
    ivorfs_result = ffi.new("IVoRFS_Result*")
    ivorfs_result.length = n_constraints
    p_result = ffi.cast("double *", result.ctypes.data)
    ivorfs_result.result = p_result
    ivorfs_result.ivrfs_error = lib.NO_IVORFIXEDSUM_ERROR
    ivorfs_result.itp_error = lib.NO_ITP_ERROR
    lib.ivorfixedsum(ivorfs_result, n_constraints, total, p_lc, p_uc, p_config)
    if ivorfs_result.ivrfs_error != lib.NO_IVORFIXEDSUM_ERROR:
        raise CFSAError(f'Received error code {ivorfs_result.ivrfs_error}-{ivorfs_result.itp_error}')
    return result


def cfsa(n: int, total: float=1.0, lower_constraints: Optional[Sequence[float]]=None,
         upper_constraints: Optional[Sequence[float]]=None, config: CFSAConfig=None):
    """
    Analytical form of ConvolutionalFixedSum.

    Note that this is a bit more rough than the numeric version at the moment. If you
    give invalid input, it's likely to produce an impenetrable exception. This will be
    improved in future versions.

    Args:
        n: Number of constraints.
        total: Total value to allocate. Defaults to 1.0. Probably should be positive.
        lower_constraints: Optional sequence of length n describing lower constraints. Defaults to all 0.
        upper_constraints: Optional sequence of length n describing upper constraints. Defaults to all total.
        config: Optional CFSA object describing configuration - including random seeds.

    Returns:
        A vector of length n, sampling uniformly from the described area.
    """
    return ivorfixedsum(n, total, lower_constraints, upper_constraints, config)


@dataclass
class CFSADistribution:
    """Analytical CFS distribution, binding to the _cfsa CFFI module.

    NOTE: The old name, IVoRFixedSum is used in a number of places. Please ignore this :)
    """
    n_constraints: int
    total: float
    lc: Optional[Iterable[float]] = None
    uc: Optional[Iterable[float]] = None

    def __post_init__(self):
        self.conf = CFSAConfig()
        self.ivorfs_vc = ffi.new("IVoRFS_VC*")
        lib.IVoRFixedSum_init(self.ivorfs_vc, self.n_constraints)
        if self.lc:
            self.lc = np.asarray(self.lc, dtype=np.float64)
        else:
            self.lc = np.zeros(self.n_constraints, dtype=np.float64)
        self.p_lc = ffi.cast("double *", self.lc.ctypes.data)
        if self.uc:
            self.uc = np.asarray(self.uc, dtype=np.float64)
            self.p_uc = ffi.cast("double *", self.uc.ctypes.data)
        else:
            self.p_uc = ffi.NULL
        lib.IVoRFixedSum_update(self.ivorfs_vc, self.n_constraints, self.p_lc, self.p_uc, self.total, self.conf.ivorfs_config)

    def __del__(self):
        lib.IVoRFixedSum_uninit(self.ivorfs_vc)

    def cdf(self, x):
        return lib.IVoRFixedSum_cdf(self.ivorfs_vc, x)

    def icdf(self, x):
        itp_res = ffi.new("ITP_Result*")
        res = lib.IVoRFixedSum_inverse_cdf_with_itp_error(self.ivorfs_vc, x, itp_res)
        if itp_res.err_code != 0:
            raise ValueError(f"ITP returned error code {itp_res.err_code}")
        return res

    def cprint(self):
        return lib.IVoRFixedSum_print(self.ivorfs_vc)

