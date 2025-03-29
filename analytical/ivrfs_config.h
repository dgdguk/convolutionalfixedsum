/**
ivrfs_config.h
==============

@brief IVoRFS Configuration struct.
@copyright David Griffin <dgdguk@gmail.com>, 2024. This file is licensed under the 3-Clause BSD License.
*/

#ifndef __IVRFS_CONFIG_H
#define __IVRFS_CONFIG_H

#include "pluggable_rand.h"
#include "itp.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// Struct representing a configuration for IVoRFixedSum. Wherever a IVoRFS_Config is given, NULL can be used which provides a default config.
typedef struct _IVoRFS_Config {
    /// Precision with which to find roots. Defaults to 1e-10.
    double epsilon;
    /// An RNG to generate random values in the range [0, 1]. Defaults to xoroshiro256+, seeded with current time. Call pluggable_rand_seed(NULL, x) to seed the default RNG with value x.
    PluggableRNG* rf;
    /// Hyperparameters to the ITP algorithm. Should normally be set to NULL, which provides sane defaults.
    ITP_Config* itp_config;
    /// If True, scale epsilon by range of interval.
    bool relative_epsilon;
    /// If relative_epsilon is set, this a minimum value of the epsilon passed to ITP.
    double minimum_epsilon;
} IVoRFS_Config;

#ifdef __cplusplus
}
#endif

#endif