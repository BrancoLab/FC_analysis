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

from analysis.loco.utils import get_experiment_data, get_bouts
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


# %%
all_bouts = get_bouts(datasets, only_in_center=True)

bouts_colors = dict(stationary=colorMap(0, name='tab10', vmin=-2, vmax=7),
                    slow=colorMap(1, name='tab10', vmin=-2, vmax=7),
                    running=colorMap(2, name='tab10', vmin=-2, vmax=7),
                    left_turn=colorMap(3, name='tab10', vmin=-2, vmax=7),
                    right_turn=colorMap(4, name='tab10', vmin=-2, vmax=7),

)

# %%
f, axarr = plt.subplots(ncols=4, nrows=5, figsize=(20, 10))

for i, (state, bouts) in enumerate(all_bouts['CNO'].items()):
    # bouts = bouts.loc[bouts.duration < 500]
    axarr[i, 0].hist(bouts.duration, bins=50, color=bouts_colors[state],
                        density=True, label=state, alpha=.25)
    axarr[i, 1].hist(bouts.distance, bins=50, color=bouts_colors[state],
                        density=True, label=state, alpha=.25)

    axarr[i, 2].hist(bouts.ang_displ, bins=50, color=bouts_colors[state],
                        density=True, label=state, alpha=.25)

    axarr[i, 3].hist(bouts.abs_ang_displ, bins=50, color=bouts_colors[state],
                        density=True, label=state, alpha=.25)

    axarr[i, 0].set(title=state + '  duration')
    axarr[i, 1].set(title='distance')
    axarr[i, 2].set(title='ang displ')
    axarr[i, 3].set(title='abs displ')

f.tight_layout()



# %%
f, axarr = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(15, 10), sharey=False)



for mouse in mice['CNO']:
    saline_left = all_bouts['SAL']['left_turn'].loc[ all_bouts['SAL']['left_turn'].mouse == mouse]
    saline_right = all_bouts['SAL']['right_turn'].loc[ all_bouts['SAL']['right_turn'].mouse == mouse]
    cno_left = all_bouts['CNO']['left_turn'].loc[ all_bouts['CNO']['left_turn'].mouse == mouse]
    cno_right = all_bouts['CNO']['right_turn'].loc[ all_bouts['CNO']['right_turn'].mouse == mouse]

    sal_left_rev = saline_left.abs_ang_displ.sum()/360
    sal_right_rev = saline_right.abs_ang_displ.sum()/360

    cno_left_rev = cno_left.abs_ang_displ.sum()/360
    cno_right_rev = cno_right.abs_ang_displ.sum()/360

    # print(f"\n\nMouse: {mouse}\n")
    # print(f"SAL, left rev {round(sal_left_rev, 3)} - right rev {round(sal_right_rev, 3)}\n" +
    #         f"      avg left {round(sal_left_rev/len(saline_left), 3)} right {round(sal_right_rev/len(saline_right), 3)}\n"+
    #         f"      # left turns {len(saline_left)}" + 
    #         f"      # right turns {len(saline_right)}")
    # print(f"CNO, left rev {round(cno_left_rev, 3)} - right rev {round(cno_right_rev, 3)}\n" +
    #         f"      avg left {round(cno_left_rev/len(cno_left), 3)} right {round(cno_right_rev/len(cno_right), 3)}\n"+
    #         f"      # left turns {len(cno_left)}" + 
    #         f"      # right turns {len(cno_right)}")


    axarr[0, 0].plot([0, 1], [sal_left_rev, sal_right_rev], 'o', ls='--',
                                ms=20, lw=6,
                                color = mice_colors['SAL'][mouse], label=mouse)
    axarr[1, 0].plot([0, 1], [cno_left_rev, cno_right_rev], 'o', ls='--',
                                ms=20, lw=6,
                                color = mice_colors['CNO'][mouse], label=mouse)
    
    axarr[0, 1].plot([0, 1], [sal_left_rev/len(saline_left), sal_right_rev/len(saline_right)], 'o', ls='--',
                                ms=20, lw=6,
                                color = mice_colors['SAL'][mouse], label=mouse)
    axarr[1, 1].plot([0, 1], [cno_left_rev/len(cno_left), cno_right_rev/len(cno_right)], 'o', ls='--',
                                ms=20, lw=6,
                                color = mice_colors['CNO'][mouse], label=mouse)
    
    axarr[0, 2].plot([0, 1], [len(saline_left), len(saline_right)], 'o', ls='--',
                                ms=20, lw=6,
                                color = mice_colors['SAL'][mouse], label=mouse)
    axarr[1, 2].plot([0, 1], [len(cno_left), len(cno_right)], 'o', ls='--',
                                ms=20, lw=6,
                                color = mice_colors['CNO'][mouse], label=mouse)

axarr[0, 0].set(title='Number of revolutions', xlim=[-.2, 1.2], ylim=[0, 30])
axarr[1, 0].set(xticks=[0, 1], xticklabels=['left', 'right'], ylim=[0, 30])
axarr[0, 1].set(title='Avg bout turn', ylim=[0, .1])
axarr[1, 1].set(xticks=[0, 1], xticklabels=['left', 'right'], ylim=[0, .1])
axarr[0, 2].set(title='# of turn bouts', ylim=[0, 350])
axarr[1, 2].set(xticks=[0, 1], xticklabels=['left', 'right'], ylim=[0, 350])
f.tight_layout()





# %%
f, axarr = plt.subplots(nrows=len(all_bouts['CNO'].keys()),  figsize=(15, 30))


for i, (state, bouts) in enumerate(all_bouts['CNO'].items()):

    idxs = choices(bouts.index.values, k=100)

    for n, idx in enumerate(idxs):
        bout = bouts.iloc[idx]

        x = bout.x - bout.x[0]
        y = bout.y - bout.y[0]
        theta = np.radians(np.nanmean(bout.orientation[:5]))

        mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        xy_hat = np.array([mtx.dot(np.array([[x_i],[y_i]])) for x_i,y_i in zip(x,y)])
        x_hat = xy_hat[:, 0]

        y_hat = xy_hat[:, 1]
        


        axarr[i].plot(x_hat, y_hat, color=bouts_colors[state], lw=2)
    
    axarr[i].set(title=state)


# %%
