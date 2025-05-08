"""
convolutionalfixedsum
*********************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license

Python module containing two implementations of the CFS algorithm, numeric (cfs) and analytical (cfsa).



Notes: This initial version matches the version used in the paper
"ConvolutionalFixedSum: Uniformly Generating Random Values with a Fixed Sum Subject to Arbitrary Constraints"
by David Griffin and Rob Davis, published at RTAS 2025. Future versions will improve upon this, for example,
by adding full documentation.

This version currently uses an old name, IVoRSFixedSum, which was later renamed to ConvolutionalFixedSum.
This will be fixed in a later version.
"""

from .cfsa import cfsa as cfsa, CFSAConfig as CFSAConfig
from .cfsvr import cfs as cfsn, cfsd as cfs_debug


