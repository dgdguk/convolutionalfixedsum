ConvolutionalFixedSum
=====================

ConvolutionalFixedSum is an algorithm for generating vectors of random numbers such that:

1. The values of the vector sum to a given total U
2. Given a vector of upper constraints, each element of the returned vector is less than or equal to its corresponding upper bound
3. Given a vector of lower constraints, each element of the returned vector is greater or equal to than its corresponding lower bound
4. The distribution of the vectors in the space defined by the constraints is uniform.

This algorithm was developed when the authors found that their prior work, the
[Dirichlet-Rescale Algorithm (DRS)](https://github.com/dgdguk/drs), did not, in fact generate values uniformly.
As such, ConvolutionalFixedSum supercedes the DRS algorithm.

Initial Version
===============

Please note, this initial version corresponds to the code for a paper currently awaiting publication.
A future version before publication will include full documentation.

Usage
=====

Two implementations of ConvolutionalFixedSum are provided: an analytical method, `cfsa`, which scales
exponentially with the length of the vector $n$ and is subject to floating point error, and the recommended numerical
approximation `cfs` which scales polynomially with $n$. `cfsa` can be useful for $n \leq 15$, while `cfs`
should work well for larger `n`. See the paper for a full discussion on this aspect.

To build the analytical CFS module, run `build_cfsa.py`. Future versions will have wheel support.

Citation
========

Following the publication of the paper, full citation information will be provided below.