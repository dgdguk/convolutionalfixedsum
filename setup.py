"""
setup.py
********

:copyright: David Griffin <dgdguk@gmail.com> (2025)
:license:  BSD-3-Clause license

Setup file for convolutionalfixedsym.

CFFI does not support pyproject.toml, so this is necessary for now.

In addition, CFFI doesn't appear to be compatible with the build package; it has no facility
to copy header files across to the build environment. So this also has a workaround where all
.h files are labeled as data files to ensure they make it to the build environment.
"""

from setuptools import setup

setup(
    package_data={"": ["*.h"]},
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["convolutionalfixedsum/build_cfsa.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"]
)