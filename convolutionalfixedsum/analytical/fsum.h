/**
fsum.h
======

Implements the Kahan-Babushka-Neumaier Floating Point Summation algorithm
But also provides access to the internal data structures, so it's possible to do
useful things like partial sums

@file fsum.h
@brief Implementation of the Kahan-Babushka-Neumaier Floating Point Summation algorithm.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#ifndef __FSUM_H
#define __FSUM_H

#ifdef __cplusplus
extern "C"
{
#endif

/// Struct for FSumData
typedef struct {
    /// Current sum
    double sum;
    /// Current compensation term
    double c;
} FSumData;

/// Reset an FSumData
void fsum_reset(FSumData* data);
/// Copy one FSumData to another
void fsum_copy(FSumData* target, FSumData* src);
/// Add x to the sum
void fsum_step(FSumData* data, const double x);
/// Add all the values in the given array to the sum
void fsum_partial(FSumData* data, unsigned int len, const double* x);
/// Subtract all the values in the given array to the sum
void fsub_partial(FSumData* data, unsigned int len, const double* x);
/// Get the current result from an FSumData
double fsum_result(FSumData* data);
/// Sum all values in the given array
double fsum(const unsigned int length, const double* input);

#ifdef __cplusplus
}
#endif

#endif
