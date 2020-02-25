# %%
# Imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from collections import namedtuple
from tqdm import tqdm
from random import choices
import seaborn as sns

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import plot_shaded_withline, ball_and_errorbar
from fcutils.plotting.colors import *
from fcutils.plotting.colors import colorMap, desaturate_color
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.maths.stats import percentile_range
from fcutils.file_io.utils import check_create_folder
from fcutils.maths.filtering import line_smoother
from fcutils.objects import flatten_list

from behaviour.plots.tracking_plots import plot_tracking_2d_trace, plot_tracking_2d_heatmap, plot_tracking_2d_scatter
from behaviour.utilities.signals import get_times_signal_high_and_low

from analysis.loco.utils import get_experiment_data, get_bouts, get_center_bouts
from analysis.misc.paths import output_fld


# %%
# --------------------------------- Variables -------------------------------- #
experiment = 'Circarena'
subexperiment = 'baseline'
cno_subexperiment = 'dreadds_sc_to_grn'
use_mouse = None
only_in_center = None

if use_mouse is not None:
    save_fld = os.path.join(output_fld, 'bouts_analysis', use_mouse)
else:
    save_fld = output_fld
check_create_folder(save_fld)

# ----------------------------------- Vars ----------------------------------- #
fps=60
keep_min = 60


# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 350



# %%
# -------------------------------- Fetch data -------------------------------- #
print("Fetching baseline")
baseline = get_experiment_data(experiment = experiment, subexperiment=subexperiment, 
                mouse = use_mouse, injected=None, center=center, radius=radius,  
                only_in_center=only_in_center, keep_min=60, fps=60)


print("Fetching CNO")
cno = get_experiment_data(experiment = experiment, subexperiment=cno_subexperiment, 
                mouse = use_mouse, injected='CNO', center=center, radius=radius,  
                only_in_center=only_in_center, keep_min=60, fps=60)


print("Fetching SAL")
sal = get_experiment_data(experiment = experiment, subexperiment=cno_subexperiment, 
                mouse = use_mouse, injected='SAL', center=center, radius=radius,  
                only_in_center=only_in_center, keep_min=60, fps=60)


# Prepare some vars
datasets = dict(baseline=baseline, CNO=cno, SAL=sal)
mice = {k:list(v.keys()) for k,v in datasets.items()}
colors = dict(baseline=[.2, .8, .2], 
            CNO=[.8, .2, .2], 
            SAL=[.2, .2, .8])
cmaps = dict(baseline='Greens', CNO='Reds', SAL='Blues')
mice_colors = {k:{m:colorMap(i, cmaps[k], vmin=-4, vmax=len(mice)+2) for i,m in enumerate(mice[k])} for k in datasets.keys()}
states = ['left_turn', 'right_turn', 'locomotion_0', 'locomotion_1', 'locomotion_2', 'locomotion_3']


state_colors = {'left_turn': salmon,
                'right_turn': lilla,
                'locomotion_0': colorMap(0, 'Greens', vmin=-2, vmax=6),
                'locomotion_1': colorMap(1, 'Greens', vmin=-2, vmax=6),
                'locomotion_2': colorMap(2, 'Greens', vmin=-2, vmax=6),
                'locomotion_3': colorMap(3, 'Greens', vmin=-2, vmax=6),
                'locomotion_4': colorMap(3, 'Greens', vmin=-2, vmax=6)}
center_bouts = get_center_bouts(datasets)


# %%
for n, mouse in enumerate(mice['CNO']): 
    f, axarr = create_figure(subplots=True, ncols=3, nrows=2, figsize=(30, 20))

    sal_bouts = center_bouts['SAL'].loc[center_bouts['SAL'].mouse == mouse]
    cno_bouts = center_bouts['CNO'].loc[center_bouts['CNO'].mouse == mouse]

    turns = {}
    for dn, (dataset, bouts) in enumerate(zip(['CNO', 'SAL'], [cno_bouts, sal_bouts])):
        bturns, speeds, angvels = [], [], []
        allspeeds, allangvels = [], []
        for i, bout in  bouts.iterrows():
            if bout.duration <60: continue
            avel = bout.ang_vel
            if np.sum(np.abs(avel)) < 25: continue
            tot_right = np.sum(avel[avel < 0])
            tot_left = -np.sum(avel[avel > 0])

            bturns.append((tot_left - tot_right)/(tot_left + tot_right))

            x, y = bout.x-bout.x[0], bout.y-bout.y[0]
            theta = np.radians(bout.orientation[0]+180)

            mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            xy = np.array([x, y])
            xy_hat = mtx.dot(xy)
            x_hat = xy_hat[0, :].ravel()
            y_hat = xy_hat[1, :].ravel()

            axarr[dn].plot(x_hat, y_hat, color=mice_colors[dataset][mouse], alpha=.75)

            if dn == 0:
                speed = bout.speed
            else:
                speed = -bout.speed
            allspeeds.extend(list(speed))
            allangvels.extend(list(bout.ang_vel))
            speeds.append(np.nanmean(speed))
            angvels.append(np.nanmean(bout.ang_vel))

            axarr[4].scatter(np.nanmean(speed), np.nanmean(bout.ang_vel), color=desaturate_color(mice_colors[dataset][mouse]),
                                    s=50, alpha=1)
   

        turns[dataset] = np.array(bturns)
        axarr[2].hist(turns[dataset], color=mice_colors[dataset][mouse], 
                        bins=10, histtype='stepfilled', alpha=.35, density=True)
        axarr[2].hist(turns[dataset], color=mice_colors[dataset][mouse], 
                        bins=10, histtype='step', alpha=1, lw=4, density=True)

        if dataset == 'CNO':
            cmap = 'Reds'
        else:
            cmap = 'Blues'
        sns.kdeplot(allspeeds, allangvels, color=mice_colors[dataset][mouse], shade=True, cmap=cmap,
                                ax=axarr[3], shade_lowest=False, alpha=.6, zorder=-1, label=dataset)
        axarr[4].scatter(np.median(speeds), np.median(angvels), color=mice_colors[dataset][mouse],
                                s=350, alpha=1, ec='k', lw=2, zorder=99)
 
        
        sns.kdeplot(speeds, angvels, color=mice_colors[dataset][mouse], shade=True, cmap=cmap,
                                ax=axarr[4], shade_lowest=False, alpha=.6, zorder=-1, label=dataset)

        if dataset == 'CNO':
            plot_kde(ax=axarr[5], data=speeds, color=mice_colors[dataset][mouse], label=dataset, kde_kwargs={'bw':.3})
        else:
            plot_kde(ax=axarr[5], data=-np.array(speeds), color=mice_colors[dataset][mouse], label=dataset, kde_kwargs={'bw':.3})

    axarr[0].set(title=f'{mouse} - center arena bouts', xlim=[-700, 700], ylim=[-700, 700])
    axarr[1].set(xlim=[-700, 700], ylim=[-700, 700])
    axarr[2].set(title='$\\frac{\\theta_L - \\theta_R}{\\theta_L + \\theta_R}$', ylabel='density',
                    xticks=[-1, 0, 1], xlabel='$\\frac{\\theta_L - \\theta_R}{\\theta_L + \\theta_R}$')

    axarr[3].set(title='Frame by frame ang vel and speed', xlabel='speed', ylabel='angular velocity',
                        ylim=[-.75, .75])
    axarr[4].set(title='Bout mean ang vel and speed', xlabel='speed', ylabel='angular velocity',
                        ylim=[-1.25, 1.25], xlim=[-10, 10])
    axarr[4].axhline(0, color='k', alpha=.5, lw=4, ls='--')
    axarr[4].axvline(0, color='k', alpha=.5, lw=4, ls='--')
    axarr[4].legend()

    axarr[3].axhline(0, color='k', alpha=.5, lw=4, ls='--')
    axarr[3].axvline(0, color='k', alpha=.5, lw=4, ls='--')
    axarr[3].legend()

    axarr[5].set(title='Avg bout speed distribution', xlabel='speed', ylabel='density')
    axarr[5].legend()

    clean_axes(f)
    # f.tight_layout()

    save_figure(f, os.path.join(output_fld, f'{mouse}_bouts_summary'))
    # break

# %%