"""
single_input
****

:author: Jordan Sun <zhuoran.sun@wustl.edu> (2025)
:license:  BSD-3-Clause license

Tests the fix for single input and same constraint case in both CFSVR and CFSA to avoid exceptions.
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

n = 4
total = 1.0
lower_constraints = [0.1, 0.2, 0.3, 0.4]
upper_constraints = lower_constraints

def test_cfsvr_same_constraint():
    result = convolutionalfixedsum.cfsn(n, total=total, lower_constraints=lower_constraints, upper_constraints=upper_constraints)
    assert result == lower_constraints

def test_cfsa_same_constraint():
    result = convolutionalfixedsum.cfsa(n, total=total, lower_constraints=lower_constraints, upper_constraints=upper_constraints)
    assert result == lower_constraints