/*
fsum.c
======

Implements the Kahan-Babushka-Neumaier Floating Point Summation algorithm
But also provides access to the internal data structures, so it's possible to do
useful things like partial sums

@file fsum.c
@brief Implementation of the Kahan-Babushka-Neumaier Floating Point Summation algorithm.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#include <math.h>
#include "fsum.h"

void fsum_reset(FSumData* data){
    // Reset and FSumData structure
    data->sum = 0.0;
    data->c = 0.0;
}

void fsum_copy(FSumData* target, FSumData* src){
    // Copy an FSumData structure
    target->sum = src->sum;
    target->c = src->c;
}

void fsum_step(FSumData* data, const double x){
    // Implements the iterative step of the Kahan-Babushka-Neumaier Floating Point Summation algorithm
    double t = data->sum + x;
    if (fabs(data->sum) >= fabs(x)){ data->c += (data->sum - t) + x;}
    else { data->c += (x - t + data->sum);}
    data->sum = t;
}

void fsum_partial(FSumData* data, unsigned int len, const double* input){
    // Applies a squence of additions from the input array
    for (unsigned int i = 0; i < len; i++){
        fsum_step(data, input[i]);
    }
}

void fsub_partial(FSumData* data, unsigned int len, const double* input){
    // Applies a sequence of subtractions from the input array
    for (unsigned int i = 0; i < len; i++){
        fsum_step(data, -input[i]);
    }
}

double fsum_result(FSumData* data){
    // Get the current result from the FSumData provided
    return data->sum + data->c;
}

double fsum(const unsigned int length, const double* input){
    // Implements the Kahan-Babushka-Neumaier Floating Point Summation algorithm
    FSumData data;
    fsum_reset(&data);
    fsum_partial(&data, length, input);
    return fsum_result(&data);
}
