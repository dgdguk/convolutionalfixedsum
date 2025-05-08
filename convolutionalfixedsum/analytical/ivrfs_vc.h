/**
ivrfs_vc.h
==========

Header file for IVoRFixedSum volume ratio calculations. These are used to
calculate the Cumulative Distribution Function for the Uniform Distribution
over the intersection between a simplex representing the lower constraints,
and a simplex representing the upper constraints, which lie on a hyperplane
given by the total allocation.

All volumes are divided by the Simplex Volume Constant (\f$\sqrt{n}/(n-1)!\f$,
where \f$n\f$ is the number of constraints). X refers to the first
dimension of the problem.

@file ivrfs_vc.h
@brief IVoRFixedSum Volume Ratio Calculation, providing the CDF and ICDF for IVoRFixedSum problems.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/
#ifndef __IVoRFixedSum_H
#define __IVoRFixedSum_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include "itp.h"
#include "ivrfs_config.h"

/// IVoRFixedSum Error codes. If multiple errors encountered, use bitwise operations to decode.
enum IVoRFixedSum_Error {
    /// No error
    NO_IVORFIXEDSUM_ERROR = 0,
    /// Returned if a lower constraint is greater than or equal to an upper constraint, rendering the problem invalid.
    LOWER_CONSTRAINT_GT_UPPER_CONSTRAINT = 1,
    /// Returned if the lower constraints are above the hyperplane of the total, rendering the problem invalid
    LOWER_CONSTRAINTS_ABOVE_TOTAL = 2,
    /// Returned if the upper constraints are below the hyperplane of the total, rendering the problem invalid
    UPPER_CONSTRAINTS_BELOW_TOTAL = 4,
    /// Returned if the ITP algorithm failed for some reason
    ITP_ERROR_DETECTED = 8,
    /// Returned if required memory was not possible to be allocated
    FAILED_TO_ALLOCATE_MEMORY = 16
};

 
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
#ifdef __cplusplus
}
#endif

#endif