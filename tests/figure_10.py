"""
figure_10
*********

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

from .runtime_experiments import cfsvr_time_experiments, cfsa_time_experiments, drs_time_experiments
from .analyse_durations import plot_graph

graph_files = [
    ['cfsa-time-data.pkl', '$CFS$, analytical'],
    ['cfsvr-1000-time-data.pkl', '$CFS$, samples=1000'],
    ['cfsvr-3000-time-data.pkl', '$CFS$, samples=3000'],
    ['cfsvr-10000-time-data.pkl', '$CFS$, samples=10000'],
    ['drs-time-data.pkl', '$DRS$'],
]

# Parameters to these functions can be varied
for sample_size in [1000, 3000, 10000]:
    cfsvr_time_experiments(f'cfsvr-{sample_size}-time-data.pkl', 10, 0, sample_size)

# WARNING: CFSA is an exponential algorithm. Use the timeout parameter to set a value in seconds for the maximum
# amount of time you are prepared for a single run to take.
# ALSO: CFSA is quite sensitive to the random constraints generated, so if you set n_exps = 1, prepare for
# some variability
cfsa_time_experiments('cfsa-time-data.pkl', 10, 0, timeout=100)
drs_time_experiments('drs-time-data.pkl', 10, 0)

# Plots the graph
plot_graph(graph_files, 'figure_10.pdf')
