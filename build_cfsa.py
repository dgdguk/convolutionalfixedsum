import sys
from cffi import FFI

WITH_IVORFS_ICDF_FUNCTIONS = True

cdef = r"""

typedef struct _PluggableRNG {
    double (*generate_func)(void* state);
    void (*seed_func)(void* state, uint64_t seed);
    void (*jump_func)(void* state);
    void* state;
    bool state_is_seeded;
} PluggableRNG;

typedef struct{
    double k1;
    double k2;
    int n0;
    bool enforce_max_iter;
    unsigned int max_iter;
} ITP_Config;

enum ITP_Error {
    NO_ITP_ERROR = 0,
    A_EQUALS_B = 1,
    INVALID_K1 = 2,
    INVALID_K2 = 4,
    INVALID_N0 = 8,   
    EPSILON_ZERO = 16,
    FUNC_INTERVAL_DOES_NOT_CROSS_ZERO = 32,
    ITP_DID_NOT_CONVERGE = 64
};

typedef struct _IVoRFS_Config {
    double epsilon;
    PluggableRNG* rf;
    ITP_Config* itp_config;
    bool relative_epsilon;
    double minimum_epsilon;
} IVoRFS_Config;

enum IVoRFixedSum_Error {
    NO_IVORFIXEDSUM_ERROR = 0,
    LOWER_CONSTRAINT_GT_UPPER_CONSTRAINT = 1,
    LOWER_CONSTRAINTS_ABOVE_TOTAL = 2,
    UPPER_CONSTRAINTS_BELOW_TOTAL = 4,
    ITP_ERROR_DETECTED = 8,
    FAILED_TO_ALLOCATE_MEMORY = 16
};

typedef struct _IVoRFS_Result {
    unsigned int length;
    double* result;
    enum IVoRFixedSum_Error ivrfs_error;
    enum ITP_Error itp_error;
} IVoRFS_Result;

double pluggable_rand_generate(PluggableRNG* pluggable_rng);
void pluggable_rand_seed(PluggableRNG* pluggable_rng, uint64_t seed);
void pluggable_rand_jump(PluggableRNG* pluggable_rng);
int pluggable_rand_xoroshiro256_rng_init(PluggableRNG* pluggable_rng);
void pluggable_rand_xoroshiro256_rng_uninit(PluggableRNG* pluggable_rng);

void ivorfixedsum(IVoRFS_Result* res, const unsigned int n_constraints, const double total, const double* lower_constraints, const double* upper_constraints, const IVoRFS_Config* config);
void IVoRFS_Result_init(IVoRFS_Result* res, const unsigned int n_constraints);
void IVoRFS_Result_uninit(IVoRFS_Result* res);
void IVoRFS_Result_print(IVoRFS_Result* res);
"""

if WITH_IVORFS_ICDF_FUNCTIONS:
    cdef += r"""

enum ITP_Warnings {
    N0_IS_ZERO = 1
};

typedef struct {
    double result;
    enum ITP_Error err_code;
    enum ITP_Warnings warnings;
    double final_size_of_range;
} ITP_Result;

/// Struct describing a IVoRFS_VC problem.
typedef struct _IVoRFS_VC {
    /// Precision to solve ICDF to
    double epsilon;
    /// ITP_Config* for ITP algorithm used in ICDF
    const ITP_Config* itp_config;
    /// Dimensions of the simplex, equal to no of constraints - 1
    unsigned int dimensions;
    /// Modified upper constraints
    double* modified_upper_constraints;
    /// Error code; should be checked before use
    enum IVoRFixedSum_Error err_code;
    /// Volume of the full area
    double full_volume;
    /// Lower constraint of first variate, used for offset
    double lower_constraint_zero;
    /// Modified total, after allocating all lower constraints
    double modified_total;
    /// Minimum value for coord zero, assuming max utilization allocated to other tasks
    double coord_zero_min;
    /// Maximum value for coord zero i.e. the constraint
    double coord_zero_max;
} IVoRFS_VC;

/// Initializes a IVoRFS_VC pointer to work with problems of at most n_constraints
void IVoRFixedSum_init(IVoRFS_VC* ivrfs, const unsigned int n_constraints);
/// Updates a IVoRFS_VC* to sample from the uniform space between lower_constraints and upper_constraints, with values summing to total.
void IVoRFixedSum_update(IVoRFS_VC* ivrfs, const unsigned int n_constraints, const double* lower_constraints, const double* upper_constraints, const double total, const IVoRFS_Config* conf);
/// Deallocate a IVoRFS_VC pointer, freeing memory.
void IVoRFixedSum_uninit(IVoRFS_VC* ivrfs);
/// Pretty printer for IVoRFS_VC structures
void IVoRFixedSum_print(const IVoRFS_VC* ivrfs);
/// Takes a IVoRFS_VC and calculates the volume above x, assuming x is in the valid region
double IVoRFixedSum_volume_above(const IVoRFS_VC* ivrfs, const double x);
//double IVoRFixedSum_volume_below(const IVoRFS_VC* ivrfs, const double x);

/// Calculates the CDF of ivrfs at point x
double IVoRFixedSum_cdf(const IVoRFS_VC* ivrfs, const double x);
/// Calculates the ICDF of ivrfs at point x
double IVoRFixedSum_inverse_cdf(const IVoRFS_VC* ivrfs, const double x);
/// Calculates the CDF of ivrfs at point x, allowing access to ITP result for error checking
double IVoRFixedSum_inverse_cdf_with_itp_error(const IVoRFS_VC* ivrfs, const double x, ITP_Result* itp_res);
/// Determine if there are any subtractive simplicies present in the problem
bool IVoRFixedSum_no_subtractive_simplcies(IVoRFS_VC* ivrfs);
"""

src = """
#include <stdbool.h>
""" + cdef


sources = [
    "analytical/itp.c",
    "analytical/fsum.c",
    "analytical/pluggable_rand.c",
    "analytical/xoroshiro256sp.c",
    "analytical/ivrfs_vc.c",
    "analytical/ivorfixedsum.c"
]

if sys.platform == 'win32':
    libraries = []
elif sys.platform in ('linux', 'darwin'):
    libraries = ['m']
else:
    import warnings
    warnings.warn("Unsupported platform - assuming it's POSIX like")
    libraries = ['m']

ffibuilder = FFI()
ffibuilder.set_source("_cfsa", src, sources=sources, libraries=libraries)
ffibuilder.cdef(cdef)

ffibuilder.compile(verbose=True)