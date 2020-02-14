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
from collections import namedtuple

from fcutils.plotting.utils import create_figure, clean_axes
from fcutils.plotting.plot_elements import plot_shaded_withline
from fcutils.plotting.colors import salmon, colorMap, goldenrod
from fcutils.maths.filtering import median_filter_1d
from fcutils.maths.stats import percentile_range

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
cno_subexperiment = 'dreadds_sc_to_grn'
bpart = 'body'

# ----------------------------------- Vars ----------------------------------- #
speed_th = 1 # frames with speed > th are considered locomotion
high_speed_th=5 # frames with speed > th are considered fast locomotion


fps=60
keep_min = 60

get_tracking_speed = partial(get_tracking_speed, fps, keep_min)

# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 350



# %%
# ---------------------------------------------------------------------------- #
#                                  FETCH DATA                                  #
# ---------------------------------------------------------------------------- #
def fetch_entries(entries, bone_entries, cmap):
    dataset = namedtuple('dataset', 'tracking_data, tracking_data_in_center, tracking_data_not_in_center, \
        tracking_data_in_center_locomoting, tracking_data_not_in_center_locomoting, bouts, outbouts, mice, colors')


    tracking_data = pd.DataFrame(entries.fetch())
    bone_tracking_data = pd.DataFrame(bone_entries.fetch())
    tracking_data['body_orientation'] = bone_tracking_data['orientation']
    tracking_data['body_ang_vel'] = bone_tracking_data['angular_velocity']

    tracking_data_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min)
    tracking_data_not_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min, reverse=True)

    tracking_data_in_center_locomoting = get_tracking_locomoting(tracking_data_in_center, center, radius,  fps, keep_min, speed_th)
    tracking_data_not_in_center_locomoting = get_tracking_locomoting(tracking_data_not_in_center, center, radius,  fps, keep_min, speed_th)

    bouts = get_bouts_df(tracking_data_in_center_locomoting, fps, keep_min)
    outbouts = get_bouts_df(tracking_data_not_in_center_locomoting, fps, keep_min)

    # Prepare some more variables
    mice = list(tracking_data.mouse_id.values)
    colors = {m:colorMap(i, cmap, vmin=-3, vmax=len(mice)) \
                        for i,m in enumerate(mice)}

    return dataset(tracking_data, tracking_data_in_center, tracking_data_not_in_center, 
                tracking_data_in_center_locomoting, tracking_data_not_in_center_locomoting, 
                bouts, outbouts, mice, colors)


# ------------------------------ Fetch baseline ------------------------------ #
baseline_entries = Session * Tracking.BodyPartTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp='{bpart}'"
bone_entries = Session * Tracking.BodySegmentTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp1='neck'" & f"bp2='body'"
baseline = fetch_entries(baseline_entries, bone_entries, 'Greens')

# --------------------------------- Fetch CNO -------------------------------- #
cno_entries = Session * Session.IPinjection * Tracking.BodyPartTracking \
                & f"exp_name='{experiment}'" &  f"injected='CNO'"\
                & f"subname='{cno_subexperiment}'" & f"bp='{bpart}'"
bone_entries = Session * Session.IPinjection * Tracking.BodySegmentTracking \
                & f"exp_name='{experiment}'" &  f"injected='CNO'"\
                & f"subname='{cno_subexperiment}'" & f"bp1='neck'" & f"bp2='body'"     
cno = fetch_entries(cno_entries, bone_entries, 'Reds')

# --------------------------------- Fetch SAL -------------------------------- #
sal_entries = Session * Session.IPinjection * Tracking.BodyPartTracking \
                & f"exp_name='{experiment}'" &  f"injected='SAL'"\
                & f"subname='{cno_subexperiment}'" & f"bp='{bpart}'"
bone_entries = Session * Session.IPinjection * Tracking.BodySegmentTracking \
                & f"exp_name='{experiment}'" &  f"injected='SAL'"\
                & f"subname='{cno_subexperiment}'" & f"bp1='neck'" & f"bp2='body'" 
sal = fetch_entries(sal_entries, bone_entries, 'Blues')

print(f"Basline entries:\n{baseline_entries}\n\nCNO entries\n{cno_entries}\n\nSAL entries\n{sal_entries}")
datasets = {'baseline':baseline, 'cno':cno, 'sal':sal}

# %%
# -------------------------------- Plot bouts -------------------------------- #

centered = False

f, axarr = create_figure(subplots=True, ncols=3, figsize=(27, 9))

for d_n, (dataset, data) in enumerate(datasets.items()):
    color = list(data.colors.values())[-1]
    for i, bout in data.outbouts.iterrows():
        if not centered:
            axarr[d_n].plot(bout.x, bout.y, color=color)
        else:
            axarr[d_n].plot(bout.x-bout.x[0], bout.y-bout.y[0], color=color)
    if centered:
        circle = plt.Circle((0, 0), 75, color=[.9, .9, .9], zorder=99)
        axarr[d_n].add_artist(circle)

    axarr[d_n].set(title=dataset+f'  {len(data.outbouts)} bouts', xticks=[], yticks=[])

clean_axes(f)



# %%
# -------------------------------- HISTOGRAMS -------------------------------- #


f, axarr = plt.subplots(nrows=5, ncols=3, figsize=(24, 18))

for d_n, (dataset, data) in enumerate(datasets.items()):
    for mouse, color in data.colors.items():
        mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(data.tracking_data_in_center_locomoting, mouse)

        axarr[0, d_n].hist(speed, color=color, histtype='stepfilled', lw=2, alpha=.4,
                           label=mouse, bins=40, density=True)

        avel = np.abs(body_ang_vel)
        avel[avel > 10] = np.nan
        axarr[1, d_n].hist(avel, color=color, histtype='stepfilled', lw=2, alpha=.4,
                    label=mouse, bins=40, density=True)

    axarr[2, d_n].hist(data.outbouts.torosity, color=color, histtype='stepfilled', lw=2, alpha=.4, label=None, 
                bins=25, density=True)

    axarr[3, d_n].hist(data.outbouts.distance, color=color, histtype='stepfilled', lw=2, alpha=.4, label=None, 
                bins=30, density=True)

    axarr[4, d_n].hist([np.nansum(v) for v in data.outbouts.ang_vel.values], 
                color=color, histtype='stepfilled', lw=2, alpha=.4, label=None, 
                bins=30, density=True)

# Style axes
axarr[0, 0].set(title='BASELINE CONTROL')
axarr[0, 1].set(title='SC->GRN | CNO')
axarr[0, 2].set(title='SC->GRN | SAL')

axarr[0, 0].set(ylabel='SPEED')
axarr[1, 0].set(ylabel='abs(ANG VEL)')
axarr[2, 0].set(ylabel='BOUT TOROSITY')
axarr[3, 0].set(ylabel='BOUT DISTANCE')
axarr[4, 0].set(ylabel='COMULATIVE BOUT ANG DISPL.')

# Speed
for ax in axarr[0, :]:
    ax.set(xlim=[0, 15], ylim=[0, 0.5])
    ax.legend()

# Ang vel
for ax in axarr[1, :]:
    ax.set(xlim=[0, 5], ylim=[0, 3])
    ax.legend()

# Torosity
for ax in axarr[2, :]:
    ax.set(xlim=[1, 2], ylim=[0, 10])

# Distance
for ax in axarr[3, :]:
    ax.set(xlim=[0, 1000], ylim=[0, 0.005])

# Tot bout ang displ
for ax in axarr[4, :]:
    ax.set(xlim=[-200, 200], ylim=[0, 0.015])

clean_axes(f)



# %%
# ------------------------------- PLOT MEDIANS ------------------------------- #

f, axarr = create_figure(subplots=True, ncols=5, figsize=(24, 6))

def get_errorbars(measure):
    return np.array([measure.median-measure.low, measure.high-measure.median]).reshape(2, 1)



for d_n, (dataset, data) in enumerate(datasets.items()):
    for m_n, (mouse, color) in enumerate(data.colors.items()):
        mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(data.tracking_data_in_center_locomoting, mouse)

        x = d_n-m_n*.05
        speed_pc = percentile_range(speed, low=25, high=75)
        avel = np.abs(body_ang_vel)
        avel[avel > 10] = np.nan # ! This is because there's still tracking errors

        avel_pc = percentile_range(avel, low=25, high=75)

        axarr[0].errorbar(x, speed_pc.median, yerr= get_errorbars(speed_pc),
                            color = color)
        axarr[0].scatter(x, speed_pc.median, color=color, s=100, zorder=99)

        axarr[1].errorbar(x, avel_pc.median, yerr= get_errorbars(avel_pc),
                            color = color)
        axarr[1].scatter(x, avel_pc.median, color=color, s=100, zorder=99)

    tor_pc = percentile_range(data.outbouts.torosity, low=25, high=75)
    dist_pc = percentile_range(data.outbouts.distance, low=25, high=75)
    tot_ang_displ = [np.nansum(v) for v in data.outbouts.ang_vel.values]
    tot_ang_displ_pc = percentile_range(tot_ang_displ, low=25, high=75)

    axarr[2].errorbar(d_n, tor_pc.median, yerr= get_errorbars(tor_pc),
                            color = color )
    axarr[2].scatter(d_n, tor_pc.median, color=color, s=100, zorder=99)
     
    axarr[3].errorbar(d_n, dist_pc.median, yerr= get_errorbars(dist_pc),
                            color = color )
    axarr[3].scatter(d_n, dist_pc.median, color=color, s=100, zorder=99)

    axarr[4].errorbar(d_n, tot_ang_displ_pc.median, yerr= get_errorbars(tot_ang_displ_pc),
                            color = color )
    axarr[4].scatter(d_n, tot_ang_displ_pc.median, color=color, s=100, zorder=99)

for ax in axarr:
    ax.set(xticks=[0, 1, 2], xticklabels=datasets.keys())

axarr[0].set(title='Median speed per individual mouse', ylabel='median speed')
axarr[1].set(title='Median abs(ang vel) per individual mouse', ylabel='abs(ang vel')
axarr[2].set(title='Median torosity per bout', ylabel='median torosity')
axarr[3].set(title='Median distance per bout', ylabel='median distance')
axarr[4].set(title='Median comulative ang displacement', ylabel='median')

clean_axes(f)



# %%
