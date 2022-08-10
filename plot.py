import matplotlib as mpl
import pickle


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


def plot_dep_k(data, fpath=None):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.3)

    ax1.set_xlabel('Number of selected items (K)')
    ax1.set_ylabel('Absolute error for maximum')

    ax2.set_xlabel('Number of selected items (K)')
    ax2.set_ylabel('Absolute error for minimum')

    ax3.set_xlabel('Number of selected items (K)')
    ax3.set_ylabel('Calculation time (sec.)')

    for name, func in data.items():
        k = list(func.keys())
        t = [item['t'] for item in func.values()]
        e_min = [item['e_min'] for item in func.values()]
        e_max = [item['e_max'] for item in func.values()]

        ax1.plot(k, e_min, label=name,
            marker='o', markersize=8, linewidth=0)
        ax2.plot(k, e_max, label=name,
            marker='o', markersize=8, linewidth=0)
        ax3.plot(k, t, label=name,
            marker='o', markersize=8, linewidth=0)

    prep_ax(ax1, xlog=False, ylog=True)
    prep_ax(ax2, xlog=False, ylog=True)
    prep_ax(ax3, xlog=False, ylog=True, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()

    if leg:
        ax.legend(loc='best', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)
