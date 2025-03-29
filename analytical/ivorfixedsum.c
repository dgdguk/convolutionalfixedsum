/*
ivorfixedsum.c
==============

The main user facing functions for the IVoRFixedSum algorithm. IVoRFixedSum
produces vectors \f$\mathbf{x}\f$ of length \f$n\f$, uniformly sampled from
a distribution specified by a total \f$t\f$ and vectors of lower and upper
constraints \f$\mathbf{lc}\f$ and \f$\mathbf{uc}\f$ such that:

1. The values of the vector \f$\mathbf{x}\f$ sum to \f$t\f$ i.e. \f$\sum_{i=1}^n \mathbf{x}_i = t\f$
2. The values in the vector \f$\mathbf{x}\f$ are greater than lower constraints \f$\mathbf{lc}\f$ i.e. \f$\mathbf{x}_i \geq \mathbf{lc}_i, \forall i\f$.
3. The values in the vector \f$\mathbf{x}\f$ are less than the upper constraints \f$\mathbf{uc}\f$ i.e.  \f$\mathbf{x}_i \leq \mathbf{uc}_i, \forall i \f$.

@brief Main user facing IVoRFixedSum functions.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "ivorfixedsum.h"
#include "pluggable_rand.h"
#include "ivrfs_vc.h"
#include "ivrfs_config.h"
#include "itp.h"
#include "fsum.h"


void IVoRFS_Result_init(IVoRFS_Result* res, unsigned int length){
    if(res->result){IVoRFS_Result_uninit(res);}
    res->length = length;
    res->result = calloc(length, sizeof(double));
    res->itp_error = 0;
    res->ivrfs_error = 0;
    if(!res->result){res->ivrfs_error = FAILED_TO_ALLOCATE_MEMORY;}
}

void IVoRFS_Result_uninit(IVoRFS_Result* res){
    free(res->result);
}

void IVoRFS_Result_print(IVoRFS_Result* res){
    printf("IVoRFS_Result(");
    if (res->ivrfs_error) {printf("ivrfs_error=%d, itp_error=%d", res->ivrfs_error, res->itp_error);}
    else {
        for (unsigned int i = 0; i < res->length; i++){
            printf("%lf", res->result[i]);
            if(i != res->length-1){printf(", ");}
        }
    }
    printf(")\n");
}

const IVoRFS_Config DEFAULT_IVORFS_CONFIG = {
    .epsilon=1e-10, .rf=NULL, .itp_config=NULL,
    .relative_epsilon=true, .minimum_epsilon=1e-14
};


void ivorfs_internal(IVoRFS_Result* res, IVoRFS_VC* d, const unsigned int n_constraints, const double total, const double* lower_constraints, const double* upper_constraints, const IVoRFS_Config* config){
    if (config == NULL){config = &DEFAULT_IVORFS_CONFIG;}
    FSumData current_remainder, aux_sum;
    double rand;
    ITP_Result itp_result;
    ITP_Result_reset(&itp_result);

    // Internals of the IVoRFixedSum algorithm. Assumes that everything has been set up, and does not allocate any memory.

    fsum_reset(&current_remainder);
    fsum_step(&current_remainder, total);
    unsigned int i = 0;

    // Allocate down to 2 constraints using the IVoRFS_VC method, unless we observe no subtractive simplicies left
    for (i = 0; i < n_constraints - 2; i++){
        IVoRFixedSum_update(d, n_constraints - i, lower_constraints + i, upper_constraints + i, fsum_result(&current_remainder), config);
        //IVoRFixedSum_print(d);
        /* Disable this optimisation
        if(IVoRFixedSum_no_subtractive_simplcies(d)) // If there are no subtractive simplicies left, switch to the linear algorithm.
        {
            break;
        }
        */
        if(d->err_code){
            res->ivrfs_error = d->err_code;
            break; // Error, so quit gracefully
        }
        rand = pluggable_rand_generate(config->rf);
        res->result[i] = IVoRFixedSum_inverse_cdf_with_itp_error(d, rand, &itp_result);
        if(itp_result.err_code){
            res->itp_error = itp_result.err_code;
            res->ivrfs_error = ITP_ERROR_DETECTED;
            break;
        }  
        fsum_step(&current_remainder, -res->result[i]);
    }

    if (res->ivrfs_error){return;}

    /* Disable this optimisation
    if(IVoRFixedSum_no_subtractive_simplcies(d) && i < n_constraints - 2) // If this is true, we have to allocate some things without subtractive simplicies.
    {

        // TODO: This should be a fsub_partial
        for(unsigned int k = i; k < n_constraints; k++){
            fsum_step(&current_remainder, -lower_constraints[k]); // Work out how much utilization is actually available
        }

        for(; i < n_constraints - 2; i++){
            rand = pluggable_rand_generate(config->rf);
            res->result[i] = fsum_result(&current_remainder) * (1 - pow(1 - rand, 1.0/(n_constraints - i - 1))); // Generate with the remaining utilization
            fsum_step(&current_remainder, -res->result[i]);
            res->result[i] += lower_constraints[i]; // Add the pre-allocated lower constraint back in
        }

        // Add the utilization of the last 2 constraints back in for the final step
        fsum_step(&current_remainder, lower_constraints[n_constraints-2]);
        fsum_step(&current_remainder, lower_constraints[n_constraints-1]);
    }
    */

    // Once we're at 2 constraints, we can just directly sample from a line segment; no need to use IVoRFS_VC.
    // Instead, work out the true lower and upper bounds for the penultimate task based on bounds of both remaining tasks and allocation remaining.
    // Note: We cannot guarantee no-subtractive simplicies here, as the beginning / end of the line segment may have an implicit bound.

    fsum_copy(&aux_sum, &current_remainder);
    fsum_step(&aux_sum, -upper_constraints[n_constraints-1]);
    double lbsum_res = fsum_result(&aux_sum);
    fsum_copy(&aux_sum, &current_remainder);
    fsum_step(&aux_sum, -lower_constraints[n_constraints-1]);
    double ubsum_res = fsum_result(&aux_sum);

    double penultimate_lb = (lower_constraints[n_constraints-2] > lbsum_res) ? lower_constraints[n_constraints-2] : lbsum_res;  // Work out true lower bound for penultimate task
    double penultimate_ub = (upper_constraints[n_constraints-2] < ubsum_res) ? upper_constraints[n_constraints-2] : ubsum_res;  // Work out true upper bound for penultimate task
    res->result[n_constraints-2] = (penultimate_ub - penultimate_lb) * pluggable_rand_generate(config->rf) + penultimate_lb;  // Pick a uniform point in the valid region
    fsum_step(&current_remainder, -res->result[n_constraints-2]);    // Update remaining allocation
    res->result[n_constraints-1] = fsum_result(&current_remainder);  // Final task is just the remaining allocation
}

void ivorfixedsum(IVoRFS_Result* res, const unsigned int n_constraints, const double total, const double* lower_constraints, const double* upper_constraints, const IVoRFS_Config* config){
    // implementation of Inverse Volume Ratio Fixed Sum (IVoRFIxedSum)
    // IVoRFS_Result* res: a IVoRFS_Result to store the result in
    // unsigned int n_constraints: number of constraints
    // double total: sum of returned vector
    // double* lower_constraints: lower bounds of the returned values; if NULL, default to 0 in all values.
    // double* upper_constraints: upper bounds of the returned values; if NULL, default to total in all values.
    // const IVoRFS_Config* config: A IVoRFS_Config - controls seeds, epsilon and ITP parameters. Specify NULL for sane defaults.
    assert(res);
    unsigned int i;
    IVoRFS_VC d;
    double* lc_default = NULL;
    double* uc_default = NULL;

    IVoRFixedSum_init(&d, n_constraints);
    if(d.err_code){res->ivrfs_error = d.err_code; return;}

    // Allocate and set lower / upper constraints if they weren't specified
    if(lower_constraints == NULL){
        lc_default = calloc(n_constraints, sizeof(double));
        if(!lc_default){res->ivrfs_error = FAILED_TO_ALLOCATE_MEMORY; return;}
        lower_constraints = lc_default;
    }
    if(upper_constraints == NULL){
        uc_default = calloc(n_constraints, sizeof(double));
        if(!uc_default){res->ivrfs_error = FAILED_TO_ALLOCATE_MEMORY; return;}
        for(i = 0; i < n_constraints; i++){uc_default[i] = total;}
        upper_constraints = uc_default;
    }

    // Now things have been set up, call the actual IVoRFixedSum algorithm
    ivorfs_internal(res, &d, n_constraints, total, lower_constraints, upper_constraints, config);

    // Teardown data structures
    IVoRFixedSum_uninit(&d);
    free(uc_default);
    free(lc_default);
}
