"""
single_input
****

:author: Jordan Sun <zhuoran.sun@wustl.edu> (2025)
:license:  BSD-3-Clause license

Tests the fix for single input case in both CFSVR and CFSA to avoid exceptions.
"""

import pytest

import convolutionalfixedsum

n = 1
total = 0.5
upper_constraints = [0.7]

def test_cfsvr_single_input():
    result = convolutionalfixedsum.cfsn(n, total=total, upper_constraints=upper_constraints)
    assert result == [total]

def test_cfsa_single_input():
    result = convolutionalfixedsum.cfsa(n, total=total, upper_constraints=upper_constraints)
    assert result == [total]