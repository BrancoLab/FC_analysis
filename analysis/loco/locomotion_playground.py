# %%
# Imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from skimage.filters import threshold_otsu

from fcutils.plotting.utils import create_figure
from fcutils.plotting.plot_elements import plot_shaded_withline
from fcutils.plotting.colors import salmon, colorMap, goldenrod
from fcutils.maths.filtering import median_filter_1d

from behaviour.plots.tracking_plots import plot_tracking_2d_trace, plot_tracking_2d_heatmap, plot_tracking_2d_scatter
from behaviour.utilities.signals import get_times_signal_high_and_low
from analysis.dbase.tables import Session, Tracking
from analysis.loco.utils import get_tracking_speed, get_tracking_in_center, get_tracking_locomoting, get_bouts_df


# %%

# ---------------------------------------------------------------------------- #
#                                   VARIABLES                                  #
# ---------------------------------------------------------------------------- #
experiment = 'Circarena'
subexperiment = 'baseline'
bpart = 'body'

# ----------------------------------- Vars ----------------------------------- #
speed_th = 1 # frames with speed > th are considered locomotion
high_speed_th=5 # frames with speed > th are considered fast locomotion


fps=60
keep_min = None

get_tracking_speed = partial(get_tracking_speed, fps, keep_min)

# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 350


# ---------------------------------------------------------------------------- #
#                                  FETCH DATA                                  #
# ---------------------------------------------------------------------------- #
entries = Session * Tracking.BodyPartTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp='{bpart}'"


bone_entries = Session * Tracking.BodySegmentTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp1='neck'" & f"bp2='body'"
bone_tracking_data = pd.DataFrame(bone_entries.fetch())

tracking_data = pd.DataFrame(entries.fetch())
tracking_data['body_orientation'] = bone_tracking_data['orientation']
tracking_data['body_ang_vel'] = bone_tracking_data['angular_velocity']

tracking_data_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min)
tracking_data_in_center_locomoting = get_tracking_locomoting(tracking_data_in_center, center, radius,  fps, keep_min, speed_th)

bouts = get_bouts_df(tracking_data_in_center, fps, keep_min)


# Prepare some more variables
mice = list(tracking_data.mouse_id.values)
colors = {m:colorMap(i, 'Greens', vmin=-3, vmax=len(mice)) \
                    for i,m in enumerate(mice)}



# %%
# ---------------------------------------------------------------------------- #
#           !                    PLOTTING STARTS                               #
# ---------------------------------------------------------------------------- #


# ----------------------------- Plot 2D tracking ----------------------------- #
# Plot 2d tracking as lines and heatmap.
for i, mouse in enumerate(mice):
    mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(tracking_data_in_center_locomoting, mouse)

    f, axarr = create_figure(subplots=True, ncols=3, figsize=(27, 9))

    plot_tracking_2d_trace(mouse_tracking, ax=axarr[0],
                        line_kwargs=dict(color=colors[mouse], lw=1),  
                        ax_kwargs=dict(title=mouse))

    plot_tracking_2d_heatmap(mouse_tracking, ax=axarr[1],
                            plot_kwargs={'bins':'log'},
                            ax_kwargs={})

    plot_tracking_2d_scatter(mouse_tracking, ax=axarr[2],
                        scatter_kwargs={'c':np.abs(ang_vel), 'cmap':'Greens', 's':5, 'vmax':2.5},
                        ax_kwargs={})

    for ax in axarr:
        ax.set(xticks=[], yticks=[])
    break


# %%

# ---------------------------- x_dot vs theta_dot ---------------------------- #
f, axarr = create_figure(subplots=True, ncols=len(mice), sharey=True)
for  i, (mouse, ax) in enumerate(zip(mice, axarr)):
    mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(tracking_data_in_center_locomoting, mouse)
    # ax.scatter(speed, np.abs(ang_vel), alpha=.6, color=colors[mouse])
    ax.plot(speed, np.abs(ang_vel), alpha=.6, color=colors[mouse])

    ax.axvline(speed_th, color=salmon, lw=2, ls='--', zorder=99)
    ax.set(xlabel='speed', ylabel='abs(ang vel)', title=mouse)


# %%
# ----------------------------- Plot speed traces ---------------------------- #
f, axarr = create_figure(subplots=True, nrows=len(mice), sharex=True, sharey=True)
f2, hist_ax = create_figure(subplots=True, ncols=2)

means, stds = [], []
for i, (mouse, ax) in enumerate(zip(mice, axarr)):
    # Prep data
    mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(tracking_data_in_center_locomoting, mouse)
    time = np.arange(len(speed))
    mean_speed = np.nanmean(speed)

    # Plot the speed trace
    plot_shaded_withline(ax, time, speed, z=0, color=colors[mouse])
    ax.axhline(mean_speed, color=colors[mouse], lw=2)

    ax.set(title=mouse+' speed trace', ylabel='Speed px/frame', xlabel='frames')

    # Plot histogram and errorbar
    hist_ax[0].hist(speed, color=colors[mouse], histtype='stepfilled', lw=2, alpha=.4,
                           label=mouse, bins=60, density=True)
    hist_ax[1].hist(ang_vel, color=colors[mouse], histtype='stepfilled', lw=2, alpha=.4,
                           label=mouse, bins=80, density=True)

hist_ax[0].axvline(speed_th, color=salmon, lw=2, ls='--', label='speed th')
hist_ax[0].axvline(high_speed_th, color=goldenrod, lw=2, ls='--', label='high_speed_th')

hist_ax[0].set(title='Speed histogram', xlabel='speed px/frame', ylabel='density')
hist_ax[0].legend()

hist_ax[1].set(title='Angular velocity histogram', xlabel='speed deg/frame', ylabel='density', 
                ylim=[0, .02], xlim=[-50, 50])
hist_ax[1].legend()





# %%

# --------------------------- Plot torosity calsses -------------------------- #
bounds = [  (1, 1.05), 
            (1.05, 1.2), 
            (1.2, 1.3), 
            (1.3, 1.6), 
            (1.6, 2.0), 
            (2.0, 5.0), 
            (5.0, 100.0)]

f, axarr = plt.subplots(nrows = len(bounds), ncols=2, figsize=(10, 24))

for b, (b0, b1) in enumerate(bounds):
    turns = bouts.loc[(bouts.torosity >= b0)&(bouts.torosity < b1)]

    for i, row in turns.iterrows():
        axarr[b,0].plot(row.x, row.y)
        axarr[b, 1].scatter(row.speed, row.ang_vel, color=salmon)
    axarr[b, 0].set(title=f'{b0} < torosity < {b1}')







# %%
# --------------------- Compute time spent locomoting etc -------------------- #
# ! Thresholds


# TODO reorganize this code and check stuff

results = dict(mouse=[], time_stationary=[], tot_locomotion=[],
            slow_locomotion=[], fast_locomotion=[])

for mouse in mice:
    _, speed, ang_vel, dir_of_mvmt = get_tracking_speed(tracking_data_in_center, mouse)
    duration = len(speed)

    stationary_time = len(np.where(speed < speed_th)[0])
    locomotion_time = duration - stationary_time
    fast_locomotion_time = len(np.where(speed > high_speed_th)[0])
    slow_locomotion_time = locomotion_time - fast_locomotion_time

    results['mouse'].append(mouse)
    results['time_stationary'].append(stationary_time/duration)
    results['tot_locomotion'].append(locomotion_time/duration)
    results['slow_locomotion'].append(slow_locomotion_time/duration)
    results['fast_locomotion'].append(fast_locomotion_time/duration)
results = pd.DataFrame(results)
results


# %%
# Get average duration/distance for locomotion bout.
results = dict(mouse=[], n_bouts=[], avg_duration=[],
            avg_distance=[])

for mouse in mice:
    _, speed, ang_vel, dir_of_mvmt = get_tracking_speed(tracking_data_in_center, mouse)

    locomotion = np.zeros_like(speed)
    locomotion[speed > speed_th] = 1

    onsets, offsets = get_times_signal_high_and_low(locomotion, th=.1)

    durations, distances = [], []
    for on, off in zip(onsets, offsets):
        durations.append(off-on)
        distances.append(np.nansum(speed[on:off]))

    results['mouse'].append(mouse)
    results['n_bouts'].append(len(onsets))
    results['avg_duration'].append(np.mean(durations))
    results['avg_distance'].append(np.mean(distances))
results = pd.DataFrame(results)
results





# %%
