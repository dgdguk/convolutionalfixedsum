"""
convolutionalfixedsum
*********************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license

Python module containing two implementations of the CFS algorithm, numeric (cfsn) and analytical (cfsa).

This algorithm was presented in the paper:

"ConvolutionalFixedSum: Uniformly Generating Random Values with a Fixed Sum Subject to Arbitrary Constraints"
by David Griffin and Rob Davis, published at RTAS 2025.

Future versions will improve upon this, for example, by adding better documentation.

This version currently uses an old name, IVoRSFixedSum in some of the source code,
which was later renamed to ConvolutionalFixedSum. This will be fixed in a later version.

Example:

    from convolutionalfixedsum import cfsn
    cfsn(3, 1.0, upper_constraints=[0.7, 0.4, 0.1])
"""

from .cfsa import cfsa as cfsa, CFSAConfig as CFSAConfig
from .cfsvr import cfs as cfsn, cfsd as cfs_debug
from .dcfs import dcfsa as dcfsa, dcfsn as dcfsn, enumeration_sampling as enumeration_sampling


