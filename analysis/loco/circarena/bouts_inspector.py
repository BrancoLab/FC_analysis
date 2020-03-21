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

from fcutils.plotting.utils import create_figure, clean_axes, save_figure, set_figure_subplots_aspect
from fcutils.plotting.plot_elements import plot_shaded_withline, ball_and_errorbar, rose_plot
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

save_fld = output_fld
check_create_folder(save_fld)

# ----------------------------------- Vars ----------------------------------- #
fps=60
keep_min = 60


# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 350

high_speed_bouts = True
speed_th = 4

# %%
# -------------------------------- Fetch data -------------------------------- #
print("Fetching baseline")
baseline = get_experiment_data(experiment = experiment, subexperiment=subexperiment, 
                injected=None, center=center, radius=radius,  
                keep_min=60, fps=60)


print("Fetching CNO")
cno = get_experiment_data(experiment = experiment, subexperiment=cno_subexperiment, 
                injected='CNO', center=center, radius=radius,  
                keep_min=60, fps=60)


print("Fetching SAL")
sal = get_experiment_data(experiment = experiment, subexperiment=cno_subexperiment, 
                injected='SAL', center=center, radius=radius,  
                keep_min=60, fps=60)


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

# ---------------------------- Mouse summary plot ---------------------------- #
# Loop over each mouse
for n, mouse in enumerate(sorted(mice['CNO'])): 
    # Create figure and subplots
    f = plt.figure(figsize=(30, 15))
    raw_trackingax = f.add_subplot(2, 4, 1, frameon=True)
    cno_trackingax = f.add_subplot(2, 4, 2)
    sal_trackingax = f.add_subplot(2, 4, 3)
    tracking_axes =  [cno_trackingax, sal_trackingax]

    turnindexax = f.add_subplot(2, 4, 4)

    angvelhistax = f.add_subplot(2, 4, 5, projection='polar')
    framespeedsax = f.add_subplot(2, 4, 6)
    meanspeedsax = f.add_subplot(2, 4, 7)
    speedsax = f.add_subplot(2, 4, 8)


    # Get bouts per condition
    sal_bouts = center_bouts['SAL'].loc[center_bouts['SAL'].mouse == mouse]
    cno_bouts = center_bouts['CNO'].loc[center_bouts['CNO'].mouse == mouse]

    # Loop over each condition
    turns = {}
    for dn, (dataset, bouts) in enumerate(zip(['CNO', 'SAL'], [cno_bouts, sal_bouts])):
        # Prepare some variables
        if dataset == 'CNO':
            cmap = 'Reds'
        else:
            cmap = 'Blues'
        mouse_color = mice_colors[dataset][mouse]

        bturns, speeds, angvels = [], [], []
        allspeeds, allangvels = [], []
        allx, ally = [], []
        x_ends = []
        all_orientations = []

        # Loop over each bout in condition
        bouts_count = 0
        for i, bout in  bouts.iterrows():
            # Ignore bouts that are too slow or fast
            if high_speed_bouts:
                if np.mean(bout.speed) < speed_th: continue
            elif not high_speed_bouts:
                if np.mean(bout.speed) > speed_th: continue

            # Ignore bouts that are too short
            if bout.duration <60: continue
            avel = bout.ang_vel * fps
            bouts_count += 1

            # Get signed angular displacement
            tot_right = np.sum(avel[avel < 0])
            tot_left = -np.sum(avel[avel > 0])
            bturns.append((tot_left - tot_right)/(tot_left + tot_right))

            # Get and correct tracking
            x, y = bout.x-bout.x[0], bout.y-bout.y[0]
            theta = np.radians(bout.orientation[0]+180)

            mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            xy = np.array([x, y])
            xy_hat = mtx.dot(xy)
            x_hat = -xy_hat[0, :].ravel()
            y_hat = xy_hat[1, :].ravel()

            allx.extend(x_hat[bout.speed > 2])
            ally.extend(y_hat[bout.speed > 2])
            x_ends.append(x_hat[-1])

            # Plot tracking
            tracking_axes[dn].plot(x_hat, y_hat, color=mouse_color,  alpha=.75)
            tracking_axes[dn].scatter(x_hat[-1], y_hat[-1], color=mouse_color, edgecolor='k', zorder=99)

            # Store some more variables
            if dn == 0:
                speed = bout.speed * fps
            else:
                speed = -bout.speed * fps
            allspeeds.extend(list(speed))
            allangvels.extend(list(avel))
            speeds.append(np.nanmean(speed))
            angvels.append(np.nanmean(avel))
            all_orientations.extend(bout.orientation - bout.orientation[0])

            # Scatter trial averaged speed and ang vel
            meanspeedsax.scatter(np.nanmean(speed), np.nanmean(avel), color=desaturate_color(mouse_color),
                                    s=50, alpha=1, edgecolor=[.2, .2, .2])
   
        # Bar plot of number of trials
        raw_trackingax.bar(dn, bouts_count, color=mouse_color, fill=True, alpha=.3)
        raw_trackingax.bar(dn, bouts_count, color=mouse_color, fill=False, edgecolor=mouse_color, alpha=1, lw=4)

        # Rose plot bout avg ang vel
        rose_plot(angvelhistax, np.radians(angvels), color=mouse_color, alpha=.3,
                        edge_color=desaturate_color(mouse_color), linewidth=4,
                        theta_min=-180, thetamax=180, nbins=37, fill=True, label=dataset)

        # Plot KDE of x position at end of bout
        plot_kde(ax=tracking_axes[dn], data=x_ends, z=-650, color=mouse_color, normto=200)
            
        # Plot Turn Index histogram
        turns[dataset] = np.array(bturns)
        turnindexax.hist(turns[dataset], color=mouse_color, 
                        bins=10, histtype='stepfilled', alpha=.35, density=True)
        turnindexax.hist(turns[dataset], color=mouse_color, 
                        bins=10, histtype='step', alpha=1, lw=4, density=True)

        # Plot instantaneous speeds 2D KDE
        sns.kdeplot(allspeeds, allangvels, color=mouse_color, shade=True, cmap=cmap,
                                ax=framespeedsax, shade_lowest=False, alpha=.6, zorder=-1, label=dataset)
        framespeedsax.scatter(np.median(allspeeds), np.median(allangvels), color=mouse_color,
                        s=350, alpha=1, edgecolor='k', lw=2, zorder=99)

        # Plot trial averaged speeds 2D KDE
        meanspeedsax.scatter(np.median(speeds), np.median(angvels), color=mouse_color,
                                s=350, alpha=1, edgecolor='k', lw=2, zorder=99)
        sns.kdeplot(speeds, angvels, color=mouse_color, shade=True, cmap=cmap,
                                ax=meanspeedsax, shade_lowest=False, alpha=.6, zorder=-1, label=dataset)

        # Plot KDE of trial averaged speeds
        if dataset == 'CNO':
            plot_kde(ax=speedsax, data=speeds, color=mouse_color, label=dataset, kde_kwargs={'bw':20})
        else:
            plot_kde(ax=speedsax, data=-np.array(speeds), color=mouse_color, label=dataset, kde_kwargs={'bw':20})


    # Set axes properties
    angvelhistax.set(title='Polar histogram of instantaneous orientation delta',
                    xticklabels=[0, 45, 90, 135, 180, -135, -90, -45])
    angvelhistax.legend()


    raw_trackingax.set(title=f'{mouse} - center bouts', xticks=[0, 1], xticklabels=['CNO', 'SAL'],
                                ylabel= "# bouts")
    cno_trackingax.set(title=f'Centered CNO bouts', xlim=[-700, 700], ylim=[-700, 700], yticks=[])
    cno_trackingax.axvline(0, color='k', alpha=.5, lw=4, ls='--')

    sal_trackingax.set(title=f'Centered SAL bouts', xlim=[-700, 700], ylim=[-700, 700], yticks=[])
    sal_trackingax.axvline(0, color='k', alpha=.5, lw=4, ls='--')

    turnindexax.set(title='$\\frac{\\theta_L - \\theta_R}{\\theta_L + \\theta_R}$', ylabel='density',
                    xticks=[-1, 0, 1], xlabel='$\\frac{\\theta_L - \\theta_R}{\\theta_L + \\theta_R}$')

    framespeedsax.set(title='Frame by frame ang vel and speed', xlabel='speed (px/s)', ylabel='angular velocity (deg/s)',
                        ylim=[-65, 65], xlim=[-600, 600])
    meanspeedsax.set(title='Bout mean ang vel and speed', xlabel='speed (px/s)', ylabel='angular velocity (deg/s)',
                        ylim=[-100, 100], xlim=[-600, 600])
    meanspeedsax.axhline(0, color='k', alpha=.5, lw=4, ls='--')
    meanspeedsax.axvline(0, color='k', alpha=.5, lw=4, ls='--')
    meanspeedsax.legend()

    framespeedsax.axhline(0, color='k', alpha=.5, lw=4, ls='--')
    framespeedsax.axvline(0, color='k', alpha=.5, lw=4, ls='--')
    framespeedsax.legend()

    speedsax.set(title='Avg bout speed distribution', xlabel='speed', ylabel='density')
    speedsax.legend()

    # Clean and save figure
    set_figure_subplots_aspect(wspace=.4, hspace=.4, top=.9, bottom=.15)
    clean_axes(f)
    f.suptitle(f'{mouse} summary.', fontsize=22)
    save_figure(f, os.path.join(output_fld, f'{mouse}_highspeed_{high_speed_bouts}_bouts_summary'))
    # break

# %%



# %%
