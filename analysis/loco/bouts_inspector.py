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

from fcutils.plotting.utils import create_figure, clean_axes, save_figure
from fcutils.plotting.plot_elements import plot_shaded_withline, ball_and_errorbar
from fcutils.plotting.colors import *
from fcutils.plotting.colors import colorMap, desaturate_color
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.maths.stats import percentile_range
from fcutils.file_io.utils import check_create_folder
from fcutils.maths.filtering import line_smoother

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
mice_colors = {k:{m:colorMap(i, cmaps[k], vmin=-4, vmax=len(mice)) for i,m in enumerate(mice[k])} for k in datasets.keys()}
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
f, axarr = plt.subplots(1, 3, figsize=(20, 10))
for dn, (dataset, bouts) in enumerate(center_bouts.items()):
    turns = []
    for i, bout in  bouts.iterrows():
        if bout.duration <60: continue
        avel = bout.ang_vel
        tot_right = np.sum(avel[avel < 0])
        tot_left = -np.sum(avel[avel > 0])

        turns.append((tot_left - tot_right)/(tot_left + tot_right))

    # plot_kde(axarr[dn], data=np.array(turns), color=colors[dataset], kde_kwargs=dict(bw=.04))
    axarr[dn].hist(np.array(turns), color=colors[dataset], bins=50)
    # axarr[dn].axvline(0, color='k', lw=2, ls='--', alpha=.5)

    ball_and_errorbar(np.median(turns), -1, turns, axarr[dn], color=colors[dataset])
    axarr


# %%
f, axarr = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)

for n, mouse in enumerate(mice['CNO']): 
    sal_bouts = center_bouts['SAL'].loc[center_bouts['SAL'].mouse == mouse]
    cno_bouts = center_bouts['CNO'].loc[center_bouts['CNO'].mouse == mouse]
    
    turns = {}
    for dataset, bouts in zip(['saline', 'cno'], [sal_bouts, cno_bouts]):
        bturns = []
        for i, bout in  bouts.iterrows():
            if bout.duration <60: continue
            avel = bout.ang_vel
            if np.sum(np.abs(avel)) < 25: continue
            tot_right = np.sum(avel[avel < 0])
            tot_left = -np.sum(avel[avel > 0])

            bturns.append((tot_left - tot_right)/(tot_left + tot_right))
        turns[dataset] = np.array(bturns)

    # plot_kde(axarr[dn], data=np.array(turns), color=colors[dataset], kde_kwargs=dict(bw=.04))
    # axarr[dn].hist(np.array(turns), color=colors[dataset], bins=50)
    # axarr[dn].axvline(0, color='k', lw=2, ls='--', alpha=.5)

    # ax.plot([0, 1], [np.mean(turns['saline']), np.mean(turns['cno'])])
    axarr[n].hist(turns['saline'], color='b', alpha=.4, density=True, bins=15)
    axarr[n].hist(turns['cno'], color='r', alpha=.4, density=True, bins=15 )
    axarr[n].set(title=mouse)



# %%
