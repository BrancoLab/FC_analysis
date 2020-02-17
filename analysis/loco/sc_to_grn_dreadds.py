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

from analysis.loco.utils import get_experiment_data
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
states = ['left_turn', 'right_turn', 'running', 'stationary']


# %%  
# -------------------------- Summary statistics plot ------------------------- #

f, axarr = plt.subplots(ncols=3, nrows=4, figsize=(25, 20))

for dn, (dataset, datas) in enumerate(datasets.items()):
    for mouse, data in datas.items():
        color = mice_colors[dataset][mouse]
        tot_frames = len(data)

        # Plot time spent in each category
        x = [np.random.normal(i, .05) for i,s in enumerate(states)]
        in_states = [len(data.loc[data.state == state]) / tot_frames for state in states]
        axarr[0, dn].plot(x, in_states, 'o', ls='--', color=color, lw=2, ms=15, alpha=.75, label=mouse)

        axarr[0, dn].set(title=dataset.upper(), ylabel='time in state', xticks=[0, 1, 2, 3], 
                            xticklabels=states, ylim=[0, 1])
        axarr[0, dn].legend()


        # Plot L vs R angular displacement
        left = data.loc[data.state == 'left_turn']
        right = data.loc[data.state == 'right_turn']

        turning_time = len(left) + len(right)

        axarr[1, dn].plot([0, 1], [len(left)/turning_time, len(right)/turning_time], 
                                'o', ls='--', color=color, lw=2, ms=15, alpha=.75, label=mouse)
        axarr[1, dn].set(ylabel='time turning', xticks=[0, 1], ylim=[0.25, .75], 
                            xticklabels=['LEFT', 'RIGHT'])
        axarr[1, dn].legend()

        # Distribution of angular velocity left vs right
        axarr[2, dn].hist(left.ang_vel.values[np.abs(left.ang_vel.values) < 20], color=color, 
                        label=mouse+'_left', bins=30, alpha=.5, density=True)
        axarr[2, dn].hist(right.ang_vel.values[np.abs(right.ang_vel.values) < 20], color=desaturate_color(color, k=.2), 
                        label=mouse+'_right', bins=30, alpha=.5, density=True)

        axarr[2, dn].set(xlim=[-10, 10], xlabel='degrees/frame', ylabel='density', title='Angular Velocity')
        axarr[2, dn].legend()

        # Plot distribution of running speeds
        running = data.loc[data.state == 'running']
        axarr[3, dn].hist(running.speed.values, label=mouse, bins=30, alpha=.5, density=True, color=color)

        axarr[3, dn].set(xlim=[0, 16], ylim=[0, .3], xlabel='px/frame', ylabel='density', title='Running speed')
        axarr[3, dn].legend()


clean_axes(f)
f.tight_layout()

# %%
for dn, (dataset, datas) in enumerate(datasets.items()):
    for mouse, data in datas.items():
        left = data.loc[data.state == 'left_turn']
        right = data.loc[data.state == 'right_turn']

        print(f'{dataset} - {mouse} - left revolutions: {np.sum(left.ang_vel) /360}, right revolutions {np.sum(np.abs(right.ang_vel)) /360}')

# %%
# --------------------------------- Get bouts -------------------------------- #
all_bouts = {}
bouts_types = ['running', 'left_turn', 'right_turn']
for dn, (dataset, datas) in enumerate(datasets.items()):
    bouts = {k:[] for k in bouts_types}
    for mouse, data in datas.items():
        tot_frames = len(data)
        state = data.state.values

        for bout_type in bouts_types:
            is_state = np.zeros(tot_frames)
            is_state[state == bout_type] = 1

            onsets, offsets = get_times_signal_high_and_low(is_state, th=.1)
            if offsets[0] < onsets[0]:
                offsets = offsets[1:]

            bts = dict(start=[], end=[], speed=[], orientation=[], ang_vel=[], x=[], y=[],
                        state=[], in_center=[], mouse=[])

            for onset, offset in zip(onsets, offsets):
                if offset < onset: raise ValueError
                
                bts['start'].append(onset)
                bts['end'].append(offset)
                bts['speed'].append(data.speed.values[onset:offset])
                bts['orientation'].append(data.orientation.values[onset:offset])
                bts['ang_vel'].append(data.ang_vel.values[onset:offset])
                bts['x'].append(data.x.values[onset:offset])
                bts['y'].append(data.y.values[onset:offset])
                bts['state'].append(data.state.values[onset:offset])
                bts['in_center'].append(data.in_center.values[onset:offset])
                bts['mouse'].append(mouse)
            
            bouts[bout_type].append(pd.DataFrame(bts))
    
    all_bouts[dataset] = {k:pd.concat(b) for k,b in bouts.items()}

# %%
# -------------------------------- Bouts plots ------------------------------- #
f, axarr = plt.subplots(nrows=4, ncols=3, figsize=(25, 20))

for dn, (dataset, bouts) in enumerate(all_bouts.items()):
    color = colors[dataset]

    # Plot mean running speed
    mean_speed = [np.nanmean(b.speed) for i, b in bouts['running'].iterrows()]
    plot_kde(axarr[0, dn], data=mean_speed, color=color, label=dataset)
    axarr[0, dn].set(title='avg bout running speed', xlabel='px/frame', ylabel='density', xlim=[2, 12], ylim=[0, .4])
    axarr[0, dn].legend()


    # Plt mean distance covered
    mean_dist = [np.sum(b.speed) for i, b in bouts['running'].iterrows()]
    plot_kde(axarr[1, dn], data=mean_dist, color=color, label=dataset)
    axarr[1, dn].set(title='avg bout running distance', xlabel='pxs', ylabel='density', xlim=[0, 800], ylim=[0, .01])
    axarr[1, dn].legend()


    # Plot avg displacement L vs R
    left_avg_displ = [np.sum(b.ang_vel) for i, b in bouts['left_turn'].iterrows()]
    right_avg_displ = [np.sum(b.ang_vel) for i, b in bouts['right_turn'].iterrows()]
    plot_kde(axarr[2, dn], data=left_avg_displ, color=color, label=dataset+' left turn')
    plot_kde(axarr[2, dn], data=np.abs(right_avg_displ), color=desaturate_color(color, k=.3), label=dataset+' right turn')
    axarr[2, dn].set(title='avg bout absolute angular displacement', xlabel='degrees', ylabel='density', xlim=[0, 100], ylim=[0, .07])
    axarr[2, dn].legend()


    # Plot avg angular velocity
    left_avg_displ = [np.mean(b.ang_vel) for i, b in bouts['left_turn'].iterrows()]
    right_avg_displ = [np.mean(b.ang_vel) for i, b in bouts['right_turn'].iterrows()]

    plot_kde(axarr[3, dn], data=left_avg_displ, color=color, label=dataset+' left turn')
    plot_kde(axarr[3, dn], data=np.abs(right_avg_displ), color=desaturate_color(color, k=.3), label=dataset+' right turn')
    axarr[3, dn].set(title='avg bout absolute angular velocity', xlabel='degrees/frame', ylabel='density', xlim=[0, 5], ylim=[0, 1.25])
    axarr[3, dn].legend()

clean_axes(f)
f.tight_layout()

# %%
f, ax = plt.subplots(figsize=(25, 20))

for dn, (dataset, datas) in enumerate(datasets.items()):
    for mouse, data in datas.items():
        color = mice_colors[dataset][mouse]
        tot_frames = len(data)
        time = np.arange(tot_frames)

        state = np.zeros(tot_frames).astype(np.float16)

        x1 = np.where(data.state == 'left_turn')[0]
        y1 = np.array([1 for _ in x1])

        x2 = np.where(data.state == 'right_turn')[0]
        y2 = np.array([0 for _ in x2])

        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])

        sort_idx = np.argsort(x)    
        x = x[sort_idx]
        y = y[sort_idx]

        sns.regplot(x, y, logistic=True, color=color, scatter=False, label=mouse)
ax.legend()

# %%
f, axarr = plt.subplots(ncols = 3, figsize=(25, 20))


for dn, (dataset, bouts) in enumerate(all_bouts.items()):
    color = colors[dataset]
    for i, (btype, bts) in enumerate(bouts.items()):
        starts = bts.start.values
        
        plot_kde(axarr[dn], data=starts, label=btype)

for ax in axarr: ax.legend()


# %%
# TODO fix bouts plotter
centered = False

f, axarr = create_figure(subplots=True, ncols=3, figsize=(27, 9))

for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue

    color = list(data.colors.values())[-1]

    if use_bouts == 'outbouts':
        bouts = data.outbouts
    elif use_bouts == 'centerbouts':
        bouts = data.centerbouts
    else:
        bouts = data.bouts

    for i, bout in bouts.iterrows():
        if not centered:
            axarr[d_n].plot(bout.x, bout.y, color=color)
        else:
            axarr[d_n].plot(bout.x-bout.x[0], bout.y-bout.y[0], color=color)
    if centered:
        circle = plt.Circle((0, 0), 75, color=[.9, .9, .9], zorder=99)
        axarr[d_n].add_artist(circle)

    axarr[d_n].set(title=dataset+f'  {len(bouts)} bouts', xticks=[], yticks=[])

clean_axes(f)
