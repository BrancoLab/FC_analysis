import matplotlib.pyplot as plt
import os
import numpy as np


def make_legend(ax, c1=[0.1, .1, .1], c2=[0.8, 0.8, 0.8], changefont=False):
    """
    Make a legend with background color c1, edge color c2 and optionally a user selected font size
    """
    if not changefont:
        legend = ax.legend(frameon=True)
    else:
        legend = ax.legend(frameon=True, prop={'size': changefont})

    frame = legend.get_frame()
    frame.set_facecolor(c1)
    frame.set_edgecolor(c2)


def save_all_open_figs(target_fld=False, name=False, format=False, exclude_number=False):
    open_figs = plt.get_fignums()

    for fnum in open_figs:
        if name:
            if not exclude_number: ttl = '{}_{}'.format(name, fnum)
            else: ttl = str(name)
        else:
            ttl = str(fnum)

        if target_fld: ttl = os.path.join(target_fld, ttl)
        if not format: ttl = '{}.{}'.format(ttl, 'svg')
        else: ttl = '{}.{}'.format(ttl, format)

        plt.figure(fnum)
        plt.savefig(ttl)


def create_figure(subplots=True, nrows=1, ncols=1, facecolor=[.1, .1, .1]):
    if not subplots:
        f = plt.figure(facecolor=facecolor)
        axarr = None
    else:
        f, axarr = plt.subplots(nrows, ncols, facecolor=facecolor)
    return f, axarr


def show(): plt.show()


def ticksrange(start, stop, step):
    return np.arange(start, stop + step, step)