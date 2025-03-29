"""
analyse_durations
*****************

:copyright: David Griffin <dgdguk@gmail.com> (2024)
:license:  BSD-3-Clause license
"""

import pickle
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt
from statistics import mean

FILES = [
    ['cfsa-time-data.pkl', '$CFS$, analytical'],
    ['cfsvr-1000-time-data.pkl', '$CFS$, samples=1000'],
    ['cfsvr-3000-time-data.pkl', '$CFS$, samples=3000'],
    ['cfsvr-10000-time-data.pkl', '$CFS$, samples=10000'],
    ['drs-time-data.pkl', '$DRS$'],
]

def plot_graph(files, outfn):
    fig, axs = plt.subplots(1, 1)
    fig.set_layout_engine('compressed')
    for file, label in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        xs = sorted(data.keys())
        ys = [mean(data[x]) for x in xs]
        plt.plot(xs, ys, label=label)
    axs.semilogy() #('log')
    axs.set_ylim(1e-3, 3e2)
    axs.set_xlabel('Number of Tasks')
    axs.set_ylabel('seconds/taskset')
    axs.legend(loc='lower right')
    #plt.tight_layout()
    plt.savefig(outfn)

if __name__ == '__main__':
    plot_graph(FILES, 'duration-graph.pdf')