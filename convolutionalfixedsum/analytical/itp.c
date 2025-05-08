/*
itp.c
=====

A C implementation of Interpolate-Truncate-Project (ITP) [1] root-finding, with some minor modifications
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

Compared to the pseduocode algorithm, the version allows:

1. interval to be in either order
2. function to be increasing or decreasing
3. a variable offset, so ITP finds \f$f(x) = offset\f$ instead of \f$f(x) = 0\f$

[1] I. F. D. Oliveira and R. H. C. Takahashi. 2020.
An Enhancement of the Bisection Method Average Performance Preserving Minmax
Optimality. ACM Trans. Math. Softw. 47, 1, Article 5 (March 2021), 24 pages.
https://doi.org/10.1145/3423597


@file itp.c
@brief A C implementation of Interpolate-Truncate-Project (ITP) [1] root-finding, with enhancements.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License. Based on the work of Oliverira and Takahashi [1].
*/

#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "itp.h"


void ITP_Result_reset(ITP_Result* res){
    res->result = 0.0;
    res->err_code = 0;
    res->warnings = 0;
    res->final_size_of_range = 0.0;
}

/// Default configuration of ITP, with hyperparameters from [1].
const ITP_Config ITP_DEFAULTS = {.k1=1.0, .k2=ITP_K2_DEFAULT, .n0=1, .enforce_max_iter=true, .max_iter=0};

ITP_Config* ITP_default_config(){
    // Returns a copy of the ITP default config that the user can modify.
    // This copy should be free'd after use. Returns NULL if memory could not be allocated.
    ITP_Config* ret = calloc(1, sizeof(ITP_Config));
    if(ret){memcpy(ret, &ITP_DEFAULTS, sizeof(ITP_Config));}
    return ret;
}

double call_func(const ITP_Function* func, const double x){
    // Calls an ITP_Function. If the ITP_Function has a pointer in data,
    // it is treated as a stateful function. Otherwise, it's an unary stateless function.
    if (func->data){
        assert(func->data_func);
        return (*func->data_func)(func->data, x);
    }
    assert(func->func);
    return (*func->func)(x);
}


unsigned int ITP_max_iter(double a, double b, const double epsilon, const ITP_Config* config){
    // Returns the max number of iterations for ITP with the given interval, epsilon and hyperparameters
    // (hyperparameters can be set to NULL for sane defaults)
    if (!config){config = &ITP_DEFAULTS;}
    return (unsigned int) ceil(log2((fabs(b - a))/(2.0*epsilon))) + config->n0;
}

void ITP_offset(ITP_Result* res, ITP_Function* func, double a, double b, double c, const double epsilon, const ITP_Config* config){
    // Implementation of the ITP root finding algorithm.
    // Modified for generality (a, b specified in any order, less restrictions on function)
    // And to find a given value for the function rather than just a root.
    // Parameters:
    // ITP_Result* res: Pointer to an ITP_Result to store the result, errors and warnings.
    // ITP_Function func: An ITP_Function struct describing the function to search for a root for.
    //                    ITP_Function provides some flexibility for configurable functions.
    // double a, double b: The interval on which to search for values of x
    // double c: Find func(x)=c
    // const double epsilon: The precision with which to find the value of x
    // const ITP_Config* config: The hyperparameters for the ITP algorithm. Specify NULL to get sane defaults.

    // Result is stored in ITP_Result->result.
    // If Result is NaN, there was an error that stopped the algorithm from executing. See the error enum for details.
    // Note that you can get result with an error code if you get lucky with your interval. The result is valid,
    // but will likely fail if the interval or function is changed.

    assert(res);

    if (!config){config = &ITP_DEFAULTS;}
    else {
        // User provided a config, so sanity check it
        if (config->k1 <= 0){res->err_code |= INVALID_K1;}
        if (config->k2 < 1){res->err_code |= INVALID_K2;}
        if (config->k2 >= ITP_MAX_K2){res->err_code |= INVALID_K2;}
        if (config->n0 < 0){res->err_code |= INVALID_N0;}
        if (config->n0 == 0){res->warnings |= N0_IS_ZERO;}
    }

    if (a == b){res->err_code |= A_EQUALS_B;}
    if (epsilon == 0){res -> err_code |= EPSILON_ZERO;}

    double swap = 0.0;
    if (a > b){swap = a; a = b; b = swap;} // Force a < b

    double y_a = call_func(func, a) - c;
    if (fabs(y_a) <= epsilon){
        res->result = a;
        return;
    }
    double y_b = call_func(func, b) - c;
    if (fabs(y_b) <= epsilon){
        res->result = b;
        return;
    }

    // We need a function such that y_a < c < y_b. If y_b < 0, we'll multiply all outputs of the function by -1 to make y_b above 0
    double direction = (y_b < 0.0 ? -1.0 : 1.0);
    y_a *= direction;
    y_b *= direction;

    // Need function over interval to cross zero, so if ya > 0, there is an error.
    if (y_a > 0.0){res->err_code |= FUNC_INTERVAL_DOES_NOT_CROSS_ZERO;}

    // If we've seen an error before we start the main algorithm, let's get out of here!
    if (res->err_code){
        res->result = nan("");
        return;
    }

    unsigned int max_iter = 0;
    if (config->enforce_max_iter){
        max_iter = config->max_iter;
        if (max_iter == 0){max_iter = ITP_max_iter(a, b, epsilon, config);}
    }

    // Set up variables for algorithm

    unsigned int n_half = (unsigned int) ceil(log2((b-a)/(2.0*epsilon)));
    unsigned int n_max = n_half + config->n0; 
    unsigned int k = 0;
    double x_f, x_half, sigma, delta, xt, r, x_itp, y_itp;

    // Main loop of algorithm
    while (b - a > 2 * epsilon){
        // Interpolate
        x_f = (y_b*a - y_a*b) / (y_b - y_a);
        // Truncate
        x_half = (a + b) / 2.0;
        sigma = (x_half - x_f < 0.0 ? -1.0 : 1.0);
        delta = config->k1 * pow(b - a, config->k2);
        if (delta <= fabs(x_half - x_f)){xt = x_f + (delta * sigma);}
        else {xt = x_half;}
        // Projection
        r = (epsilon * pow(2.0, (double)(n_max - k))) - ((b - a) / 2.0);
        if (fabs(xt - x_half) <= r){x_itp = xt;}
        else {x_itp = x_half - (r * sigma);}
        // Update Interval
        y_itp = (call_func(func, x_itp) - c) * direction;
        if (y_itp > 0.0){b = x_itp; y_b = y_itp;}
        else if (y_itp < 0.0){a = x_itp; y_a = y_itp;}
        else {a = b = x_itp;}
        if (config->enforce_max_iter){
            if(k > max_iter){
                res->err_code |= ITP_DID_NOT_CONVERGE;
                res->result = nan("");
                res->final_size_of_range = fabs(b - a);
                return;
            }
        }
        k += 1;
    }
    // Final result is the midpoint of the interval, which lies within epsilon of the true solution.
    res->result = (a + b) / 2;
    res->final_size_of_range = fabs(b - a);
    return;
}

void ITP(ITP_Result* res, ITP_Function* func, double a, double b, const double epsilon, const ITP_Config* config){
    // Implementation of ITP with a function signature close to the original paper
    ITP_offset(res, func, a, b, 0.0, epsilon, config);
}

double ITP_result_only(ITP_Function* func, double a, double b, const double epsilon, const ITP_Config* config){
    // ITP, but just return the result with no error codes.
    // You still have some error checking - if it returns NaN, there was an error and you should call
    // ITP so you can extract the error code.
    ITP_Result* res = calloc(1, sizeof(ITP_Result));
    assert(res);  // MSVC not being smart (if ITP resturns, res is not null)
    ITP(res, func, a, b, epsilon, config);
    double r = res->result;
    free(res);
    return r;
}
