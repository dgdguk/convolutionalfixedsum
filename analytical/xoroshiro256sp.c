/*
xoroshiro256sp.c
================

A lightly repackaged version of the xoroshiro256 ** and + algorithms by
David Blackman and Sebastiano Vigna (vigna@acm.org).

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

@file xoroshiro256sp.c
@brief A repackaged version of the xoroshiro256** and xoroshiro256+ algorithms by David Blackman and Sebastiano Vigna.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is placed in the public domain. Modified from the public domain work of David Blackman and Sebastiano Vigna.
*/

#include <stdint.h>
#include "xoroshiro256sp.h"

uint64_t splitmix64_next(uint64_t* x) {
	uint64_t z = (*x += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

void xoroshiro256_seed(XoroshiroState* xo, uint64_t seed){
   xo->state[0] = splitmix64_next(&seed);
   xo->state[1] = splitmix64_next(&seed);
   xo->state[2] = splitmix64_next(&seed);
   xo->state[3] = splitmix64_next(&seed);
}

static inline uint64_t xoroshiro256_rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static inline void __update_xoroshiro_state(XoroshiroState* xo){
	const uint64_t t = xo->state[1] << 17;
	xo->state[2] ^= xo->state[0];
	xo->state[3] ^= xo->state[1];
	xo->state[1] ^= xo->state[2];
	xo->state[0] ^= xo->state[3];
	xo->state[2] ^= t;
	xo->state[3] = xoroshiro256_rotl(xo->state[3], 45);
}

uint64_t xoroshiro256ss_next(XoroshiroState* xo) {
   const uint64_t result = xoroshiro256_rotl(xo->state[1] * 5, 7) * 9;
   __update_xoroshiro_state(xo);
	return result;   
}

uint64_t __xoroshiro256p_next(XoroshiroState* xo) {
	const uint64_t result = xo->state[0] + xo->state[3];
   __update_xoroshiro_state(xo);
	return result;
}

double xoroshiro256p_next(XoroshiroState* xo){
   // xoroshiro256+ function is used for doubles
   return (__xoroshiro256p_next(xo) >> 11) * 0x1.0p-53;
}

void xoroshiro256_jump(XoroshiroState* xo) {
   // Advances xo by 2^128 steps, allowing for 2^128 unique sequences of 2^128.
	static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= xo->state[0];
				s1 ^= xo->state[1];
				s2 ^= xo->state[2];
				s3 ^= xo->state[3];
			}
			xoroshiro256p_next(xo);	
		}
	xo->state[0] = s0;
	xo->state[1] = s1;
	xo->state[2] = s2;
	xo->state[3] = s3;
}
