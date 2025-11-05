ConvolutionalFixedSum
=====================

ConvolutionalFixedSum (CFS) is an algorithm for generating vectors of random numbers such that:

1. The values of the vector sum to a given total U
2. Given a vector of upper constraints, each element of the returned vector is less than or equal to its corresponding upper bound
3. Given a vector of lower constraints, each element of the returned vector is greater or equal to than its corresponding lower bound
4. The distribution of the vectors in the space defined by the constraints is uniform.

This algorithm was developed when the authors found that their prior work, the
[Dirichlet-Rescale Algorithm (DRS)](https://github.com/dgdguk/drs), did not, in fact generate values uniformly.
As such, ConvolutionalFixedSum supercedes the DRS algorithm.

This package also includes Discrete-ConvolutionalFixedSum (DCFS) which allows the specification of a discrete lattice
to sample from.

Usage
=====

Two implementations of ConvolutionalFixedSum are provided: an analytical method, `cfsa`, which scales
exponentially with the length of the vector $n$ and is subject to floating point error, and the recommended numerical
approximation `cfsn` which scales polynomially with $n$. `cfsa` can be useful for $n \leq 15$, while `cfs`
should work well for larger `n`. See the paper for a full discussion on this aspect.

Below are some examples on how to use the library.

```python
from convolutionalfixedsum import cfsa, cfsn

# Generate 3 random values which sum to 2.0, are bounded below by 0 and above by 1
cfsn(3, total=2.0) 

# Generate 3 random values which sum to 1.0, and whose upper constraints are [1.0, 0.5, 0.1] respectively
cfsn(3, upper_constraints=[1.0, 0.5, 0.1])

# Same as before, but the middle value can not be lower than 0.3
cfsn(3, lower_constraints=[0.0, 0.3, 0.0], upper_constraints=[1.0, 0.5, 0.1])

# cfsa has basically the same function signature for most uses
cfsa(3, lower_constraints=[0.0, 0.3, 0.0], upper_constraints=[1.0, 0.5, 0.1])
```

To use Discrete ConvolutionalFixedSum, use the `dcfsa` and `dcfsn` functions. These behave the same as `cfsa` and `cfsn` respectively, but with additional parameters to specify the lattice of valid solutions.

```python
from convolutionalfixedsum import dcfsa, dcfsn, enumeration_sampling

# Generate 3 random values which sum to 1.0 and are on the lattice [0.1, 0.1, 0.2] centered at [0.0, 0.0, 0.0]
dcfsa(3, lattice_spacing=[0.1, 0.1, 0.2], lattice_origin=[0.0, 0.0, 0.0])

# dcfsn has the basically the same function signature for most uses
dcfsn(3, lattice_spacing=[0.1, 0.1, 0.2], lattice_origin=[0.0, 0.0, 0.0])

# enumeration sampling enumerates all lattice points and samples from the list. It has an additional k parameter to control the number of points returned, as once enumeration has been done, additional samples are almost free
enumeration_sampling(3, k=100, lattice_spacing=[0.1, 0.1, 0.2], lattice_origin=[0.0, 0.0, 0.0])
```

Citation
========

If you wish to cite this software package, use the cite this repository feature of Github.
It can give citation data in a variety of formats.

If you wish to cite the underlying research, please cite the following paper

```bibtex
@inproceedings{Griffin2025,
  title={ConvolutionalFixedSum: Uniformly Generating Random Values with a Fixed Sum Subject to Arbitrary Constraints},
  author={Griffin, David and Davis, Robert I.},
  booktitle={31st {IEEE} Real-Time and Embedded Technology and Applications Symposium},
  year={2025},
  doi={https://doi.org/10.1109/RTAS65571.2025.00034}
}
```

If you wish to cite the research for DCFS, please cite the following paper

```bibtex
@inproceedings{Griffin2025a,
  title={Discrete ConvolutionalFixedSum},
  author={Griffin, David and Davis, Robert I.},
  booktitle={33rd International Conference on Real-Time Networks and Systems},
  year={2025},
}
```
