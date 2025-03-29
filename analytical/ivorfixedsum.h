/**
ivorfixedsum.h
==============

The main user facing functions for the IVoRFixedSum algorithm. IVoRFixedSum
produces vectors \f$\mathbf{x}\f$ of length \f$n\f$, uniformly sampled from
a distribution specified by a total \f$t\f$ and vectors of lower and upper
constraints \f$\mathbf{lc}\f$ and \f$\mathbf{uc}\f$ such that:

1. The values of the vector \f$\mathbf{x}\f$ sum to \f$t\f$ i.e. \f$\sum_{i=1}^n \mathbf{x}_i = t\f$
2. The values in the vector \f$\mathbf{x}\f$ are greater than lower constraints \f$\mathbf{lc}\f$ i.e. \f$\mathbf{x}_i \geq \mathbf{lc}_i, \forall i\f$.
3. The values in the vector \f$\mathbf{x}\f$ are less than the upper constraints \f$\mathbf{uc}\f$ i.e.  \f$\mathbf{x}_i \leq \mathbf{uc}_i, \forall i \f$.

@file ivorfixedsum.h
@brief Main user facing IVoRFixedSum functions.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#ifndef __IVORFIXEDSUM_H
#define __IVORFIXEDSUM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "ivrfs_vc.h"
#include "ivrfs_config.h"
#include "itp.h"

/// Struct representing the result of a call to IVoRFixedSum
typedef struct _IVoRFS_Result {
    /// Length of the result
    unsigned int length;
    /// Array containing the result
    double* result;
    /// Indicator of errors. If this is nonzero, an error occurred and the result is invalid.
    enum IVoRFixedSum_Error ivrfs_error;
    /// Further information in the case of an error in the ITP algorithm.
    enum ITP_Error itp_error;
} IVoRFS_Result;

/**
Call the IVoRFixedSum algorithm, storing the result in res. n_constraints (\f$n\f$) specifies the number of constraints. total specifies the total (\f$t\f$) to sum to.
lower_constraints specifies the lower constraints vector (\f$\mathbf{lc}\f$). If NULL, defaults to all 0's. upper_constraints specifies the upper constraints
vector (\f$\mathbf{uc}\f$). If NULL, defaults to all maximum. config is a IVoRFS_Config to configure the algorithm. If NULL, use sane defaults.

@brief Call the IVoRFixedSum algorithm.
*/
void ivorfixedsum(IVoRFS_Result* res, const unsigned int n_constraints, const double total, const double* lower_constraints, const double* upper_constraints, const IVoRFS_Config* config);
/// Intialize a IVoRFS_Result to hold a result for n_constraints
void IVoRFS_Result_init(IVoRFS_Result* res, const unsigned int n_constraints);
/// Free resources inside a IVoRFS_Result
void IVoRFS_Result_uninit(IVoRFS_Result* res);
/// Pretty print a IVoRFS_Result
void IVoRFS_Result_print(IVoRFS_Result* res);
/// Internals of the IVoRFixedSum algorithm, primarily for embedding into
/// other languages which may want to set these things up separately.
/// Avoids the setup/teardown of the IVoRFixedSum function as it assumes
/// that everything has been handled elsewhere. Only config can be null.
void ivorfs_internal(IVoRFS_Result* res, IVoRFS_VC* d, const unsigned int n_constraints, const double total, const double* lower_constraints, const double* upper_constraints, const IVoRFS_Config* config);

#ifdef __cplusplus
}
#endif

#endif