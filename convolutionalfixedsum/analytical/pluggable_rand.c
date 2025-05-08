/*
pluggable_rand.c
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

@file pluggable_rand.c
@brief Pluggable framework for RNGs, with a default RNG based on xoroshiro256+.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include "xoroshiro256sp.h"
#include "pluggable_rand.h"

#ifdef __linux__
#include <sys/time.h>
/// Linux get_time_for_seed function - returns time in microseconds
uint64_t get_time_for_seed(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t t = 1000000 * tv.tv_sec + tv.tv_usec;
    return t;
}
#else
#include <time.h>
/// Fallback get_time_for_seed function - returns time in seconds
uint64_t get_time_for_seed(){
    return time(NULL);
}
#endif

/// Default adapter function for xoroshiro256+ random number generation
double pluggable_rand_xoroshiro256p_gen(void* state){
    return xoroshiro256p_next((XoroshiroState*) state);
}

/// Default adapter function for seeding xoroshiro256 state seeding
void pluggable_rand_xoroshiro256_seed(void* state, uint64_t seed){
    xoroshiro256_seed((XoroshiroState*) state, seed);
}

/// Default adapter function for seeding xoroshiro256 jump
void pluggable_rand_xoroshiro256_jump(void* state){
    xoroshiro256_jump((XoroshiroState*) state);
}

/// State for the default RNG
XoroshiroState default_state;

/// Description of the default RNG
PluggableRNG default_rand_func = {
    .generate_func = *pluggable_rand_xoroshiro256p_gen,
    .seed_func = *pluggable_rand_xoroshiro256_seed,
    .jump_func = *pluggable_rand_xoroshiro256_jump,
    .state = &default_state,
    .state_is_seeded = false
};

int pluggable_rand_xoroshiro256_rng_init(PluggableRNG* pluggable_rng){
    pluggable_rng->generate_func = *pluggable_rand_xoroshiro256p_gen;
    pluggable_rng->seed_func = *pluggable_rand_xoroshiro256_seed;
    pluggable_rng->jump_func = *pluggable_rand_xoroshiro256_jump;
    pluggable_rng->state = calloc(1, sizeof(XoroshiroState));
    pluggable_rng->state_is_seeded = false;
    return (!pluggable_rng->state) ? 1 : 0;
}

void pluggable_rand_xoroshiro256_rng_uninit(PluggableRNG* pluggable_rng){
    free(pluggable_rng->state);
}

void pluggable_rand_seed(PluggableRNG* pluggable_rng, uint64_t seed){
    if (pluggable_rng == NULL){pluggable_rng = &default_rand_func;}
    (*pluggable_rng->seed_func)(pluggable_rng->state, seed);
    pluggable_rng->state_is_seeded = true;
}

double pluggable_rand_generate(PluggableRNG* pluggable_rng){
    if (pluggable_rng == NULL){pluggable_rng = &default_rand_func;}
    if (!pluggable_rng->state_is_seeded){
            pluggable_rand_seed(pluggable_rng, get_time_for_seed());
    }
    return (*pluggable_rng->generate_func)(pluggable_rng->state);
}

void pluggable_rand_jump(PluggableRNG* pluggable_rng){
    if (pluggable_rng == NULL){pluggable_rng = &default_rand_func;}
    if (!pluggable_rng->state_is_seeded){
            pluggable_rand_seed(pluggable_rng, get_time_for_seed());
    }
    (*pluggable_rng->jump_func)(pluggable_rng->state);     
}