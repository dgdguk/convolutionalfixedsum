/**
xoroshiro256sp.h
================

A lightly repackaged version of the xoroshiro256** and xoroshiro256+ algorithms
by David Blackman and Sebastiano Vigna (vigna@acm.org).

The original versions of the algorihthms can be found at
https://prng.di.unimi.it/.

Modifications are as follows:

1. Bundling in a seed function based on splitmix64 (as recommended)
2. Passing in state as a pointer rather than a static variable.
   This is slower, but more appropriate for threaded values.
3. Provides functions for uint64 and double that use appropriate methods
4. A little refactoring to remove duplicate code.

Note that it is possible to mix and match xoroshiro256** and xoroshiro256+ on the
same state.

@file xoroshiro256sp.h
@brief A repackaged version of the xoroshiro256** and xoroshiro256+ algorithms by David Blackman and Sebastiano Vigna.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is placed in the public domain. Modified from the public domain work of David Blackman and Sebastiano Vigna.
*/

#include <stdint.h>

/// State of a Xoroshiro256 PRNG
typedef struct {
    // 256-bit state vector
    uint64_t state[4];
} XoroshiroState;

/// Seed a XoroshroState with a given 64-bit seed, using an internal Splitmix64 function to expand the 64-bit seed to the 256-bit state vector
void xoroshiro256_seed(XoroshiroState* xo, uint64_t seed);

/// Generate a double between 0 and 1 using the xoroshiro256+ algorithm; this is suitable for doubles as it generates enough high quality random bits for a double.
double xoroshiro256p_next(XoroshiroState* xo);

/// Generate a uint64_t using the xoroshiro2** algirithm, which is needed as it requires 64 high quality random bits.
uint64_t xoroshiro256ss_next(XoroshiroState* xo);

/// Jump the xoroshiro256 state ahead by 2^128 steps.
void xoroshiro256_jump(XoroshiroState* xo);