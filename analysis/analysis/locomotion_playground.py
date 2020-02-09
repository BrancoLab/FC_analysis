# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fcutils.plotting.utils import create_figure, 
from fcutils.plotting.plotting_elements import plot_shaded_withline
from fcutils.plotting.colors import salmon, colorMap

from behaviour.plots.tracking_plots import plot_tracking_2d_trace, plot_tracking_2d_heatmap
from behaviour.utilities.signals import get_times_signal_high_and_low
from analysis.dbase.tables import Session, Tracking

# %%
# Get data
experiment = 'Circarena'
subexperiment = 'baseline'
bpart = 'body'

entries = Session * Tracking.BodyPartTracking & f"exp_name='{experiment}''" \
                & f"subname='{subexperiment}'" & f"bp='{bpart}'"
tracking_data = pd.DataFrame(entries.fetch())

# TODO inspect how data are organize

# Prepare some variables
mice = []
colors = {m:colorMap(i, 'tab20', vmin=0, vmax=len(mice)) \
                    for i,m in enumerate(mice)}

# ----------------------------- Useful Functions ----------------------------- #
def get_tracking_speed(tracking_data, mouse):
    # aaaaaa
    return tracking_data, speed

# %%
# ----------------------------- Few generic plots ---------------------------- #
# Plot 2d tracking as lines and heatmap.
for i, mouse in enumerate(mice):
    mouse_tracking, _ = get_tracking_speed(tracking_data, mouse)

    axarr = create_figure(subplots=True, ncols=2)

    plot_tracking_2d_trace(mouse_tracking, ax=axarr[0],
                        line_kwargs=dict(color=colors[mouse], lw=1),  
                        ax_kwargs=dict(title=mouse))

    plot_tracking_2d_heatmap(mouse_tracking, ax=axarr[1],
                            plot_kwargs={},
                            ax_kwargs={})

    for ax in axarr:
        ax.set(xticks=[], yticks=[])


# %%
# Plot speed traces, histograms etc
f, axarr = create_figure(subplots=True, nrows=len(mice), sharex=True, sharey=True)
f2, summary_axes = create_figure(subplots=True, ncols=2)

means, stds = [], []
for i, (mouse, ax) in enumerate(zip(mice, axarr)):
    # Plot speed trace
    _, speed = get_tracking_speed(tracking_data, mouse)
    time = np.arange(len(speed))

    plot_shaded_withline(ax, time, speed, z=0, color=colors[mouse])

    ax.set(title=mouse+' speed trace', ylabel='Speed px/frame', xlabel='frames ')

    # Plot histogram and errorbar
    summary_axes[0].hist(speed, color=colors[mouse], histtype='step', lw=2, alpha=.6, label=mouse)
    summary_axes[1].errorbar(i, np.nanmean(speed), yerr=np.nanstd(speed), 
                                color=colors[mouse], ls='o')

summary_axes[0].set(title='Speed histogram')
summary_axes[0].legend()


# %%
# --------------------- Compute time spent locomoting etc -------------------- #
# ! Thresholds
speed_th = 1 # frames with speed > th are considered locomotion
high_speed_th=5 # frames with speed > th are considered fast locomotion

reults = dict(mouse=[], time_stationary=[], tot_locomotion=[],
            slow_locomotion=[], fast_locomotion=[])

for mouse in mice:
    _, speed = get_tracking_speed(tracking_data, mouse)
    duration = len(speed)

    stationary_time = len(np.where(speed < th)[0])
    locomotion_time = duration - stationary_time
    fast_locomotion_time = len(np.where(speed > high_speed_th)[0])
    slow_locomotion_time = locomotion_time - fast_locomotion_time

    results['mouse'].append(mouse)
    results['time_stationary'].append(stationary_time/duration)
    results['tot_locomotion'].append(locomotion_time/duration)
    results['slow_locomotion'].append(slow_locomotion/duration)
    results['fast_locomotion'].append(fast_locomotion/duration)
results = pd.DataFrame(results)
results


# %%
# Get average duration/distance for locomotion bout.
results = dict(mouse=[], n_bouts=[], avg_duration=[],
            avg_distance=[])

for mouse in mice:
    _, speed = get_tracking_speed(tracking_data, mouse)

    locomotion = np.zeros_like(speed)
    locomotion[speed > speed_th] = 1

    onsets, offsets = get_times_signal_high_and_low(locomotion)

    durations, distances = [], []
    for on, off in zip(onsets, offsets):
        durations.append(off-on)
        distances.append(np.sum(speed[on:off]))

    results['mouse'].append(mouse)
    results['n_bouts'].append(len(onsets))
    results['avg_duration'].append(np.mean(durations))
    results['avg_distance'].append(np.mean(distances))
results = pd.DataFrame(results)
results



