/**
pluggable_rand.h
================

As the benchmark for what is considered high-quality pseudo-random
number generation may change independently of this project, this
provids a pluggable interface for random number generation.

pluggable_rand also provides a "default" random number generator, currently
using xoroshiro256+. The default can be used by specifying NULL wherever
a PluggableRNG* is used.

PluggableRNG specifies a generation function, a seed function, and some
state. On first use a PluggableRNG will be seeded with the current time in
microsends if needed. To see how to set up a PluggableRNG, look at the
default below.

@file pluggable_rand.h
@brief Pluggable framework for RNGs, with a default RNG based on xoroshiro256+.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#ifndef __PLUGGABLE_RAND_H
#define __PLUGGABLE_RAND_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdbool.h>

/// Struct representing a Random Generator
typedef struct _PluggableRNG {
    /// Function that generates new random numbers. If the state has not been seeded, it is automatically seeded with the current time in microseconds.
    double (*generate_func)(void* state);
    /// Function that seeds an RNG with a 64-bit seed
    void (*seed_func)(void* state, uint64_t seed);
    /// Function to jump RNG state ahead by an amount determined by the implementation
    void (*jump_func)(void* state);
    /// Pointer to some state 
    void* state;
    /// Boolean to determine if the state has been seeded.
    bool state_is_seeded;
} PluggableRNG;

/// Generate a random double in [0, 1] by calling the generate function of a PluggableRNG. If pluggable_rng is NULL, use the default RNG.
double pluggable_rand_generate(PluggableRNG* pluggable_rng);

/// Call the seed function for a PluggableRNG. If pluggable_rng is NULL, use the default RNG.
void pluggable_rand_seed(PluggableRNG* pluggable_rng, uint64_t seed);

/// Call the jump function for a PluggableRNG, which advances the RNG by a significant (although implementation dependant) amount
void pluggable_rand_jump(PluggableRNG* pluggable_rng);

/// Initialise a PluggableRNG to use the xoroshiro256 algorithm. Returns 1 if successful, 0 if failed.
int pluggable_rand_xoroshiro256_rng_init(PluggableRNG* pluggable_rng);

/// Free resources from a PluggableRNG using the xoroshiro256 algorithm.
void pluggable_rand_xoroshiro256_rng_uninit(PluggableRNG* pluggable_rng);

#ifdef __cplusplus
}
#endif

#endif
