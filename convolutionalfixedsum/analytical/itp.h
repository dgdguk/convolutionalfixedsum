/**
itp.h
=====

A C implementation of Interpolate-Truncate-Project (ITP) [1] root-finding,
with some minor modifications that relax some of the constraints of the
pseudocode implementation in [1].

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

Compared to the pseduocode algorithm, this version allows:

1. interval to be in either order
2. function to be increasing or decreasing
3. a variable offset, so ITP finds \f$f(x) = offset\f$ instead of \f$f(x) = 0\f$

[1] I. F. D. Oliveira and R. H. C. Takahashi. 2020.
An Enhancement of the Bisection Method Average Performance Preserving Minmax
Optimality. ACM Trans. Math. Softw. 47, 1, Article 5 (March 2021), 24 pages.
https://doi.org/10.1145/3423597


@file itp.h
@brief A C implementation of Interpolate-Truncate-Project (ITP) [1] root-finding, with enhancements.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License. Based on the work of Oliverira and Takahashi [1].
*/

#ifndef __ITP_H
#define __ITP_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <math.h>
#include <stdbool.h>

/// Max value of K2 = (1+(0.5*(1+sqrt(5)))). Clang throws a "not compile time constant" if using the formula though.
#define ITP_MAX_K2 2.618033988749895
// Default value of K2 = 0.98*ITP_MAX_K2, but MSVC complains about this not being compile time constant
#define ITP_K2_DEFAULT 2.565673308974897

/// Configuration for ITP
typedef struct{
    /// K1 hyperparameter; must be a positive real. 1 is the default value.
    double k1;
    /// K2 hyperparameter; must be in \f$[1, 1+\frac{1+\sqrt{5}}{2})\f$. \f$ 0.98\times(1+\frac{1+\sqrt{5}}{2})\f$ is the default value. 
    double k2;
    /// N0 hyperparameter; must be a non-negative integer. 1 is the default value.
    int n0;
    /// By default ITP does not have a maximum number of iterations, but relies on a proof of convergence. However, numerical instability can break this. If true, enforces a maximum number of iterations. Defaults to true.
    bool enforce_max_iter;
    /// If enforce_max_iter is true, the maximum number of iterations allowed before a non-convergence error. If 0, the theoretical max is used. Defaults to 0.
    unsigned int max_iter;
} ITP_Config;

/// Error codes for ITP
enum ITP_Error {
    /// No error
    NO_ITP_ERROR = 0,
    /// Interval must span some value
    A_EQUALS_B = 1,
    /// K1 must be a positive real
    INVALID_K1 = 2,
    /// K2 must be in \f$[1, 1+\frac{1+\sqrt{5}}{2}]\f$
    INVALID_K2 = 4,
    /// N0 must be a non-negative integer
    INVALID_N0 = 8,   
    /// Desired accuracy cannot be zero  
    EPSILON_ZERO = 16,
    /// func(a) and func(b) must be on opposite sides of zero
    FUNC_INTERVAL_DOES_NOT_CROSS_ZERO = 32,
    /// ITP did not converge; this may be due to the accuracy or stability of the function supplied
    ITP_DID_NOT_CONVERGE = 64
};

/// Warning codes for ITP
enum ITP_Warnings {
    /// If N0 is set to zero, then there is a possibility of numerical instability.
    N0_IS_ZERO = 1
};

/// Struct for the result, errors, warnings and diagnostics of ITP
typedef struct {
    /// The result found, where func(result) = offset. If there was a fatal error, this is NaN.
    double result;
    /// An error code. If this is nonzero, there was a problem.
    enum ITP_Error err_code;
    /// A warning code. If this is nonzero, then for some inputs you may have a problem.
    enum ITP_Warnings warnings;
    /// Final size of the range. If you get error ITP_DID_NOT_CONVERGE, you can use this to determine how close you were to convergence and if it may be worth relaxing your criteria.
    double final_size_of_range;
} ITP_Result;

/// Struct reprenting a callback for ITP. Two function signatures are supported. If data is supplied, the data_func signature will be used. Otherwise, the func signature.
typedef struct {
    union {
        // Signature for a function that requires no extra data (e.g. a standard C function)
        double (*func)(const double);
         // Signature for a function that requires some extra data, which is specified in the data pointer (e.g. a C function that is configurable).
        double (*data_func)(const void*, const double);
    };
    // Data supplied to data_func. If NULL, the func pointer is used instead.
    const void* data;
} ITP_Function;

/// Returns a copy of the default config which can then be modified if required
ITP_Config* ITP_default_config();

/// Apply ITP to the given problem, finding \f$x \in [a, b]\f$ such that \f$|f(x) - c| < \epsilon\f$. 
void ITP_offset(ITP_Result* res, ITP_Function* func, double a, double b, double c, const double epsilon, const ITP_Config* config);
/// Traditional form of ITP, finding \f$x \in [a, b]\f$ such that \f$|f(x)| < \epsilon\f$. 
void ITP(ITP_Result* res, ITP_Function* func, double a, double b, const double epsilon, const ITP_Config* config);
/// ITP but just returns a result
double ITP_result_only(ITP_Function* func, double a, double b, const double epsilon, const ITP_Config* config);
/// Returns the max number of iterations ITP will take for the given problem
unsigned int ITP_max_iter(double a, double b, const double epsilon, const ITP_Config* config);
/// Initialize / reset an ITP result
void ITP_Result_reset(ITP_Result* res);

#ifdef __cplusplus
}
#endif

#endif