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
from fcutils.plotting.colors import salmon, colorMap, goldenrod, desaturate_color
from fcutils.plotting.plot_distributions import plot_kde
from fcutils.maths.filtering import median_filter_1d
from fcutils.maths.stats import percentile_range
from fcutils.file_io.utils import check_create_folder

from behaviour.plots.tracking_plots import plot_tracking_2d_trace, plot_tracking_2d_heatmap, plot_tracking_2d_scatter
from behaviour.utilities.signals import get_times_signal_high_and_low

from analysis.dbase.tables import Session, Tracking
from analysis.loco.utils import get_tracking_speed, fetch_entries
from analysis.misc.paths import output_fld


# %%
# --------------------------------- Variables -------------------------------- #
experiment = 'Circarena'
subexperiment = 'baseline'
cno_subexperiment = 'dreadds_sc_to_grn'
bpart = 'body'
use_mouse = None

if use_mouse is not None:
    save_fld = os.path.join(output_fld, 'bouts_analysis', use_mouse)
else:
    save_fld = output_fld
check_create_folder(save_fld)

# ----------------------------------- Vars ----------------------------------- #
speed_th = 1 # frames with speed > th are considered locomotion

fps=60
keep_min = 60

get_tracking_speed = partial(get_tracking_speed, fps, keep_min)

# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 400



# %%
# ---------------------------------------------------------------------------- #
#                                  FETCH DATA                                  #
# ---------------------------------------------------------------------------- #

# ------------------------------ Fetch baseline ------------------------------ #
print("Fetching baseline")
baseline_entries = Session * Tracking.BodyPartTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp='{bpart}'"
bone_entries = Session * Tracking.BodySegmentTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp1='neck'" & f"bp2='body'"

if use_mouse is not None:
    baseline_entries = baseline_entries & f"mouse_id='{use_mouse}'"
    bone_entries = bone_entries & f"mouse_id='{use_mouse}'"

baseline = fetch_entries(baseline_entries, bone_entries, 'Greens', center, radius,  
                    fps, keep_min, speed_th)

# --------------------------------- Fetch CNO -------------------------------- #
print("Fetching CNO")
cno_entries = Session * Session.IPinjection * Tracking.BodyPartTracking \
                & f"exp_name='{experiment}'" &  f"injected='CNO'"\
                & f"subname='{cno_subexperiment}'" & f"bp='{bpart}'"
bone_entries = Session * Session.IPinjection * Tracking.BodySegmentTracking \
                & f"exp_name='{experiment}'" &  f"injected='CNO'"\
                & f"subname='{cno_subexperiment}'" & f"bp1='neck'" & f"bp2='body'"    
cno_neck_entries = Session * Session.IPinjection * Tracking.BodyPartTracking \
                & f"exp_name='{experiment}'" &  f"injected='CNO'"\
                & f"subname='{cno_subexperiment}'" & f"bp='neck'" 

if use_mouse is not None:
    cno_entries = cno_entries & f"mouse_id='{use_mouse}'"
    bone_entries = bone_entries & f"mouse_id='{use_mouse}'"
    cno_neck_entries = cno_neck_entries & f"mouse_id='{use_mouse}'"

cno = fetch_entries(cno_entries, bone_entries, 'Reds', center, radius,  
                    fps, keep_min, speed_th, neck_entries=cno_neck_entries)


# --------------------------------- Fetch SAL -------------------------------- #
print("Fetching SAL")
sal_entries = Session * Session.IPinjection * Tracking.BodyPartTracking \
                & f"exp_name='{experiment}'" &  f"injected='SAL'"\
                & f"subname='{cno_subexperiment}'" & f"bp='{bpart}'"
bone_entries = Session * Session.IPinjection * Tracking.BodySegmentTracking \
                & f"exp_name='{experiment}'" &  f"injected='SAL'"\
                & f"subname='{cno_subexperiment}'" & f"bp1='neck'" & f"bp2='body'"
if use_mouse is not None:
    sal_entries = sal_entries & f"mouse_id='{use_mouse}'"
    bone_entries = bone_entries & f"mouse_id='{use_mouse}'"

sal = fetch_entries(sal_entries, bone_entries, 'Blues', center, radius,  
                    fps, keep_min, speed_th)

print(f"Basline entries:\n{baseline_entries}\n\nCNO entries\n{cno_entries}\n\nSAL entries\n{sal_entries}")
datasets = {'baseline':baseline, 'cno':cno, 'sal':sal}


# %%
use_bouts = 'centerbouts'

# %%
# -------------------------------- Plot bouts -------------------------------- #

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

save_figure(f, os.path.join(save_fld, 'bouts'))

# %%
# -------------------------------- HISTOGRAMS -------------------------------- #
f, axarr = plt.subplots(nrows=5, ncols=3, figsize=(24, 18))

for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue
    for mouse, color in data.colors.items():
        mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(data.tracking_data_in_center, mouse)

        # Speed histogram
        axarr[0, d_n].hist(speed, color=color, histtype='stepfilled', lw=2, alpha=.4,
                           label=mouse, bins=40, density=True)

        # Angular velocity
        avel = ang_vel
        avel[avel > 5] = np.nan
        avel[avel < -5] = np.nan
        avel[avel == 0] = np.nan

        axarr[1, d_n].hist(avel, color=color, histtype='stepfilled', lw=2, alpha=.4,
                    label=mouse, bins=40, density=True)
        axarr[1, d_n].axvline(0, color=desaturate_color(color, k=.7), ls='--', lw=3, zorder=99)


    if use_bouts == 'outbouts':
        bouts = data.outbouts
    elif use_bouts == 'centerbouts':
        bouts = data.centerbouts
    else:
        bouts = data.bouts

    # Remove outliers to make histogram easier to plot
    bouts = bouts.loc[bouts.torosity < 2]

    # bouts torosity
    axarr[2, d_n].hist(bouts.torosity, color=color, histtype='stepfilled', lw=2, alpha=.4, label=None, 
                bins=20, density=True)

    # bouts distance
    axarr[3, d_n].hist(bouts.distance, color=color, histtype='stepfilled', lw=2, alpha=.4, label=None, 
                bins=15, density=True)

    # bouts comulative angular displacement
    axarr[4, d_n].hist([np.nansum(v) for v in bouts.ang_vel.values], 
                color=color, histtype='stepfilled', lw=2, alpha=.4, label=None, 
                bins=20, density=True)
    axarr[4, d_n].axvline(0, color=desaturate_color(color, k=.7), ls='--', lw=3, zorder=99)

# Style axes
if use_bouts == 'outbouts':
    axarr[0, 0].set(title='BASELINE CONTROL [out bouts]')
elif use_bouts == 'centerbouts':
    axarr[0, 0].set(title='BASELINE CONTROL [center bouts]')
else:
    axarr[0, 0].set(title='BASELINE CONTROL [bouts]')

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
    ax.set(xlim=[-4, 4], ylim=[0, .6])
    ax.legend()

# Torosity
for ax in axarr[2, :]:
    ax.set(xlim=[1, 3], ylim=[0, 5])

# Distance
for ax in axarr[3, :]:
    ax.set(xlim=[0, 2000], ylim=[0, 0.005])

# Tot bout ang displ
for ax in axarr[4, :]:
    ax.set(xlim=[-200, 200], ylim=[0, 0.015])

clean_axes(f)

save_figure(f, os.path.join(save_fld, 'histograms'))


# %%
# ------------------------------- PLOT MEDIANS ------------------------------- #
f, axarr = create_figure(subplots=True, ncols=5, figsize=(24, 6))

def get_errorbars(measure):
    return np.array([measure.median-measure.low, measure.high-measure.median]).reshape(2, 1)


for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue
    for m_n, (mouse, color) in enumerate(data.colors.items()):
        mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                         = get_tracking_speed(data.tracking_data_in_center, mouse)

        # prep vars
        x = d_n-m_n*.05
        speed_pc = percentile_range(speed, low=25, high=75)
        avel = np.abs(body_ang_vel)
        avel[avel > 10] = np.nan 

        avel_pc = percentile_range(avel, low=25, high=75)

        # speed
        ball_and_errorbar(x, speed_pc.median, axarr[0], yerr= get_errorbars(speed_pc), color=color)

        # ang vel
        ball_and_errorbar(x, avel_pc.median, axarr[1], yerr= get_errorbars(avel_pc), color=color)

    # prepr bouts vars
    if use_bouts == 'outbouts':
        bouts = data.outbouts
    elif use_bouts == 'centerbouts':
        bouts = data.centerbouts
    else:
        bouts = data.bouts

    tor_pc = percentile_range(bouts.torosity, low=25, high=75)
    dist_pc = percentile_range(bouts.distance, low=25, high=75)
    tot_ang_displ = [np.nansum(v) for v in bouts.ang_vel.values]
    tot_ang_displ_pc = percentile_range(tot_ang_displ, low=25, high=75)

    # torosity
    ball_and_errorbar(d_n, tor_pc.median, axarr[2], yerr= get_errorbars(tor_pc), color=color)

    # distance
    ball_and_errorbar(d_n, dist_pc.median, axarr[3], yerr= get_errorbars(dist_pc), color=color)

    # comulative angular displacement
    ball_and_errorbar(d_n, tot_ang_displ_pc.median, axarr[4], yerr= get_errorbars(tot_ang_displ_pc), color=color)
    axarr[4].axhline(0, lw=3, ls='--', color=[.7, .7, .7])

# Style axes
for ax in axarr:
    ax.set(xticks=[0, 1, 2], xticklabels=datasets.keys())

if use_bouts == 'outbouts':
    axarr[0].set(title='Median speed per individual mouse | out bouts', ylabel='median speed')
elif use_bouts == 'centerbouts':
    axarr[0].set(title='Median speed per individual mouse | center bouts', ylabel='median speed')
else:
    axarr[0].set(title='Median speed per individual mouse | bouts', ylabel='median speed')
axarr[1].set(title='Median abs(ang vel) per individual mouse', ylabel='abs(ang vel')
axarr[2].set(title='Median torosity per bout', ylabel='median torosity')
axarr[3].set(title='Median distance per bout', ylabel='median distance')
axarr[4].set(title='Median comulative ang displacement', ylabel='median')

f.tight_layout()
clean_axes(f)

save_figure(f, os.path.join(save_fld, 'medians'))

# %%
# --------------------- Look at high tor vs low tor bouts -------------------- #
f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))

centered=False

for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue
    if use_bouts == 'outbouts':
        bouts = data.outbouts.sort_values('torosity')
    elif use_bouts == 'centerbouts':
        bouts = data.centerbouts.sort_values('torosity')
    else:
        bouts = data.bouts.sort_values('torosity')

    colors = list(data.colors.values())
    quarter_bouts = int(len(bouts)/4)

    for n, bb in enumerate([bouts[:quarter_bouts], bouts[-quarter_bouts:]]):
        for (i, bout) in bb.iterrows():
            if not centered:
                axarr[n, d_n].plot(bout.x, bout.y, color=colors[0])
            else:
                axarr[n, d_n].plot(bout.x-bout.x[0], bout.y-bout.y[0], color=colors[-1])

    axarr[0, d_n].set(title=dataset+'  | LOW torosity', xticks=[], yticks=[])
    axarr[1, d_n].set(title='HIGH torosity', xticks=[], yticks=[])

clean_axes(f)
save_figure(f, os.path.join(save_fld, 'high_vs_low_tor'))

# %%
# ----------------- Look at curvature of high torosity bouts ----------------- #
f, axarr = plt.subplots(ncols=3,figsize=(18, 6), sharey=True, sharex=True)

mean_thetas = {d:[] for d in datasets.keys()}
for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue
    if use_bouts == 'outbouts':
        bouts = data.outbouts.sort_values('torosity')
    elif use_bouts == 'centerbouts':
        bouts = data.centerbouts.sort_values('torosity')
    else:
        bouts = data.bouts.sort_values('torosity')

    color = list(data.colors.values())[0]
    quarter_bouts = int(len(bouts)/4)
    
    all_thetas = []
    for i, bout in bouts[-quarter_bouts:].iterrows():
        theta = np.unwrap(bout.body_orientation - bout.body_orientation[0])
        axarr[d_n].plot(theta, color=color, alpha=.5)
        all_thetas.extend(list(theta))

        median_theta = np.nanmean(theta)
        if not np.isnan(median_theta):
            mean_thetas[dataset].append(median_theta)

    # Plot KDE
    plot_kde(axarr[d_n], data=all_thetas, vertical=True, normto=-300, z=-100, color=color)

    axarr[d_n].axhline(0, lw=2, ls='--', color=[.7, .7, .7], alpha=.4, zorder=-1)
    if d_n == 0:
        ylab = 'degrees'
        ttl = ' bouts body orientation'
    else:
        ylab, ttl = None, ''
    axarr[d_n].set(title=dataset.upper()+ttl, xlabel='framen n', 
                        ylabel=ylab, ylim=[-200, 200], xlim=[-500, 2000])
clean_axes(f)
save_figure(f, os.path.join(save_fld, 'high_tor_curvature'))

# %%
# Distribution of mean theta
f, ax = plt.subplots(figsize=(10, 10))

for i, (dataset, mean_theta) in enumerate(mean_thetas.items()):
    if not datasets[dataset].mice: continue
    color = list(datasets[dataset].colors.values())[0]
    ax.scatter(np.random.normal(i, .05, size=len(mean_theta)), mean_theta, color=color)
    ax.scatter(i, np.nanmean(mean_theta), color=desaturate_color(color), edgecolor='k', s=300, zorder=99)

ax.axhline(0, lw=2, ls=':', color=[.5, .5, .5], zorder=-1)

from scipy import stats
res = stats.ttest_ind(mean_thetas['cno'], mean_thetas['sal'])
ax.set(title = f't values {res[0]}, p values {res[1]}')

save_figure(f, os.path.join(save_fld, 'mean_thetas distr'))


# %%
# ------------------------- Plot all bouts in figures ------------------------ #
dest_fld = os.path.join(save_fld, 'all_bouts')
check_create_folder(dest_fld)


for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue
    print(f"\nSaving all bouts plots {dataset}")
    color = list(datasets[dataset].colors.values())[0]

    if use_bouts == 'outbouts':
        bouts = data.outbouts.sort_values('torosity')
    elif use_bouts == 'centerbouts':
        bouts = data.centerbouts.sort_values('torosity')
    else:
        bouts = data.bouts.sort_values('torosity')

    fld = os.path.join(dest_fld, dataset)
    check_create_folder(fld)

    f = None
    n_bouts_in_figure = 0
    for n, (i, bout) in tqdm(enumerate(bouts.iterrows())):
        if n_bouts_in_figure == 10:
            if f is not None:
                save_figure(f, os.path.join(fld, f'bouts_{n}'))
                plt.close(f)

            f, ax = create_figure(subplots=False, figsize=(12, 12))
            ax.set(title=f'bouts starting at {n}')
            clean_axes(f)

            n_bouts_in_figure = 0
        

        ax.plot(bout.x, bout.y, color=color)
        n_bouts_in_figure += 1






# %%
# # ------------------ Plot inner and outer bouts for figures ----------------- #

# f, axarr = create_figure(subplots=True, ncols=3, figsize=(27, 9))

# for d_n, (dataset, data) in enumerate(datasets.items()):
#     if not data.mice: continue
#     color = list(data.colors.values())[-1]

#     for i, bout in data.outbouts.iterrows():
#             axarr[d_n].plot(bout.x, bout.y, color=color, alpha=.3)

#     for i, bout in data.bouts.iterrows():
#             axarr[d_n].plot(bout.x, bout.y, color=color)

#     axarr[d_n].set(title=dataset.upper(), xticks=[], yticks=[])

# clean_axes(f)

