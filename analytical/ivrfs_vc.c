/*
ivrfs_vc.c
==========

Header file for IVoRFixedSum volume ratio calculations. These are used to
calculate the Cumulative Distribution Function for the Uniform Distribution
over the intersection between a simplex representing the lower constraints,
and a simplex representing the upper constraints, which lie on a hyperplane
given by the total allocation.

All volumes are divided by the Simplex Volume Constant (\f$\sqrt{n}/(n-1)!\f$,
where \f$n\f$ is the number of constraints). X refers to the first
dimension of the problem.

@file ivrfs_vc.c
@brief IVoRFixedSum Volume Ratio Calculation, providing the CDF and ICDF for IVorFixedSum problems.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ivrfs_vc.h"
#include "itp.h"
#include "fsum.h"
#include "ivrfs_config.h"

void IVoRFixedSum_init(IVoRFS_VC* ivrfs, const unsigned int n_constraints){
    ivrfs->err_code = 0;
    ivrfs->modified_upper_constraints = calloc(n_constraints, sizeof(double));
    if(!ivrfs->modified_upper_constraints){ivrfs->err_code |= FAILED_TO_ALLOCATE_MEMORY; return;}
}

void IVoRFixedSum_uninit(IVoRFS_VC* ivrfs){
    free(ivrfs->modified_upper_constraints);
}

void IVoRFixedSum_update(
    IVoRFS_VC* ivrfs, const unsigned int n_constraints, 
    const double* lower_constraints, const double* upper_constraints,
    const double total, const IVoRFS_Config* conf)
{

    ITP_Config* itp_config = conf->itp_config;
    // Basic parameters
    ivrfs->dimensions = n_constraints - 1;
    ivrfs->itp_config = itp_config;

    FSumData fd;       // FSumData for whatever fsum we're working on

    // Calculate modified total = total - sum(lower_constraints)
    fsum_reset(&fd);
    fsum_step(&fd, total);
    fsub_partial(&fd, n_constraints, lower_constraints);
    ivrfs->modified_total = fsum_result(&fd);

    // Calculate the modified upper constraints
    for(unsigned int i = 0; i < n_constraints; i++){
        if(lower_constraints[i] >= upper_constraints[i]){
            ivrfs->err_code |= LOWER_CONSTRAINT_GT_UPPER_CONSTRAINT;
        }
        ivrfs->modified_upper_constraints[i] = upper_constraints[i] - lower_constraints[i];
    }

    ivrfs->coord_zero_max = ivrfs->modified_upper_constraints[0];

    // Store the offset for the CDF of the first variate
    ivrfs->lower_constraint_zero = lower_constraints[0];

    // Calculate the minimum value for the first coordinate, assuming all available allocation goes
    // to other coordinates;

    fsum_reset(&fd);
    fsum_step(&fd, total);
    fsub_partial(&fd, n_constraints - 1, ivrfs->modified_upper_constraints + 1);
    ivrfs->coord_zero_min = fsum_result(&fd);
    if(ivrfs->coord_zero_min < 0.0){ivrfs->coord_zero_min = 0;}

    // Set epsilong value
    if(conf->relative_epsilon){
        ivrfs->epsilon = (ivrfs->coord_zero_max - ivrfs->coord_zero_min) * conf->epsilon;
        if (ivrfs->epsilon < conf->minimum_epsilon){ivrfs->epsilon = conf->minimum_epsilon;}
    }
    else{
        ivrfs->epsilon = conf->epsilon;
    }
    // Calculate the full volume of the valid region
    ivrfs->full_volume = IVoRFixedSum_volume_above(ivrfs, ivrfs->coord_zero_max);
    return;    
}

bool IVoRFixedSum_no_subtractive_simplcies(IVoRFS_VC* ivrfs){
    return 0; // MOD: Not ready for this optimisation yet.
    //return ivrfs->n_subtractive_simplex_offsets == 0 && ivrfs->bottom_subtractive_simplex_volume == 0.0;
}

void IVoRFixedSum_print(const IVoRFS_VC* ivrfs){
    unsigned int i;
    printf("IVoRFS_VC([");
    if(ivrfs->err_code){printf("err_code=%d)\n", ivrfs->err_code); return;}
    printf("dimensions=%u, full_volume=%lf, modified_upper_constraints=[",
    ivrfs->dimensions, ivrfs->full_volume);
    for(i = 0; i < ivrfs->dimensions+1; i++){
        printf("%lf", ivrfs->modified_upper_constraints[i]);
        if(i < ivrfs->dimensions){printf(", ");}
    }
    printf("], modified_total=%lf, coord_zero_min=%lf, lower_constraint_zero=%lf)\n",
           ivrfs->modified_total, ivrfs->coord_zero_min, ivrfs->lower_constraint_zero);
}

void cfs_analytical_conv_i(unsigned int indx, const double total, const unsigned int n_constraints,
                           const double* constraints, const unsigned int dim,
                           const unsigned int n_applied_constraints,
                           FSumData* constraint_sum, FSumData* sum_terms){
     if (fsum_result(constraint_sum) >= total){}
     else if (indx < n_constraints){
        FSumData constraint_sum2;
        fsum_copy(&constraint_sum2, constraint_sum);
        fsum_step(&constraint_sum2, constraints[indx]);
        cfs_analytical_conv_i(indx + 1, total, n_constraints, constraints, dim,
                              n_applied_constraints + 1, &constraint_sum2, sum_terms);
        cfs_analytical_conv_i(indx + 1, total, n_constraints, constraints, dim,
                              n_applied_constraints, constraint_sum, sum_terms);
     }
     else{
        fsum_step(sum_terms, pow(-1, n_applied_constraints) * pow(total - fsum_result(constraint_sum), dim));
     }
}

double IVoRFixedSum_volume_above(const IVoRFS_VC* ivrfs, const double x){
    double* constraints = ivrfs->modified_upper_constraints;
    constraints[0] = x;
    FSumData constraint_sum;
    FSumData sum_terms;
    fsum_reset(&constraint_sum);
    fsum_reset(&sum_terms);
    cfs_analytical_conv_i(0, ivrfs->modified_total, ivrfs->dimensions+1, constraints, ivrfs->dimensions, 0,
                          &constraint_sum, &sum_terms);
    return fsum_result(&sum_terms);
}


double IVoRFixedSum_translated_cdf(const IVoRFS_VC* ivrfs, const double x){
    assert(ivrfs->err_code == 0);
    if(x <= ivrfs->coord_zero_min){return 0.0;}
    else if(x >= ivrfs->coord_zero_max){return 1.0;}
    else{
        return (IVoRFixedSum_volume_above(ivrfs, x) / ivrfs->full_volume);
    }
}

double IVoRFixedSum_cdf(const IVoRFS_VC* ivrfs, const double x){
    return IVoRFixedSum_translated_cdf(ivrfs, x + ivrfs->lower_constraint_zero);
}

double IVoRFixedSum_cdf_itp(const void* void_ivrfs, const double x){
    // Does the necessary casting for ITPs configurable function.
    return IVoRFixedSum_translated_cdf((const IVoRFS_VC*) void_ivrfs, x);
}

double IVoRFixedSum_inverse_cdf_with_itp_error(const IVoRFS_VC* ivrfs, const double x, ITP_Result* itp_res){
    ITP_Function itp_func = {.data_func=&IVoRFixedSum_cdf_itp, .data=ivrfs};
    ITP_offset(itp_res, &itp_func, ivrfs->coord_zero_min, ivrfs->coord_zero_max, x, ivrfs->epsilon, ivrfs->itp_config);
    // Add lower_constraint_zero to the result to "recover" from the translation
    return itp_res->result + ivrfs->lower_constraint_zero;;
}

double IVoRFixedSum_inverse_cdf(const IVoRFS_VC* ivrfs, const double x){
    ITP_Result itp_res;
    ITP_Result_reset(&itp_res);
    double res = IVoRFixedSum_inverse_cdf_with_itp_error(ivrfs, x, &itp_res);
    assert(itp_res.err_code == 0);
    return res;
}
