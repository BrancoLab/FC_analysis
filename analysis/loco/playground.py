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
from fcutils.maths.stats import percentile_range
from fcutils.file_io.utils import check_create_folder
from fcutils.maths.filtering import median_filter_1d

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

# ----------------------------------- Vars ----------------------------------- #
speed_th = 1 # frames with speed > th are considered locomotion

fps=60
keep_min = 60


# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 400



# %%
# Fetch data
print("Fetching baseline")
baseline_entries = Session  * Session.IPinjection * Tracking.BodyPartTracking & f"exp_name='{experiment}'" \
                & f"subname='{cno_subexperiment}'" & f"mouse_id='CA826'" &  f"injected='CNO'"
bone_entries = Session  * Session.IPinjection * Tracking.BodySegmentTracking & f"exp_name='{experiment}'" \
                & f"subname='{cno_subexperiment}'" & f"mouse_id='CA826'" &  f"injected='CNO'"

bparts = pd.DataFrame(baseline_entries.fetch())
bones = pd.DataFrame(bone_entries.fetch())

head = bones.loc[(bones.bp1=='neck')&(bones.bp2=='body')].iloc[0]
body = bones.loc[(bones.bp1=='body')&(bones.bp2=='tail')].iloc[0]

# %%
# Get clean angular velocity
head_theta_dot = np.concatenate([[0], np.degrees(np.diff(np.unwrap(np.radians(np.nan_to_num(head.orientation)))))])
body_theta_dot = np.concatenate([[0], np.degrees(np.diff(np.unwrap(np.radians(np.nan_to_num(body.orientation)))))])
mean_theta_dot = median_filter_1d(np.mean([head_theta_dot, body_theta_dot], axis=0))


# %%
# Get clear speed
speeds = np.vstack([s for s in bparts.loc[bparts.bp != 'snout'].speed.values])

mean_x_dot = median_filter_1d(np.nanmean(speeds, axis=0))



# %%
turners = np.where((np.abs(mean_theta_dot) > mean_x_dot/4)&(np.abs(mean_theta_dot) > 0.5))[0]
runners = np.where((np.abs(mean_theta_dot) < mean_x_dot/4)&(mean_x_dot>2))[0]

turns = np.array([mean_x_dot[turners], mean_theta_dot[turners]])
runs = np.array([mean_x_dot[runners], mean_theta_dot[runners]])

state = np.zeros_like(mean_x_dot)
state[turners] = 1
state[runners] = 2
# %%
f, ax = plt.subplots(figsize=(10, 10))
ax.scatter(turns[0, :50000], np.abs(turns[1, :50000]), s=10, alpha=.05)
ax.scatter(runs[0, :50000], np.abs(runs[1, :50000]), s=10, alpha=.05)
ax.set(xlim=[0, 12], ylim=[0, 5])
ax.plot([0, 5], [0, 5])

ax.plot([0, 6], [0, 1.5])


# %%
plt.plot(state)

# %%
keep = np.where((mean_x_dot > .5)&(mean_theta_dot != 0))[0]
f, ax = plt.subplots(figsize=(10, 10))
ax.hexbin(mean_x_dot[keep], mean_theta_dot[keep], bins='log')



# %%
from sklearn import preprocessing
standardised_x_dot = preprocessing.scale(mean_x_dot)
standardised_theta_dot = preprocessing.scale(mean_theta_dot)

f, ax = plt.subplots(figsize=(10, 10))
ax.hexbin(standardised_x_dot, standardised_theta_dot, bins='log')

dataset = pd.DataFrame(dict(x=np.nan_to_num(standardised_x_dot), y=np.nan_to_num(standardised_theta_dot)))
# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
res = kmeans.fit(dataset)

y_kmeans = kmeans.fit_predict(dataset)
dataset['cluster'] = y_kmeans

# %%
f, ax = plt.subplots(figsize=(10, 10))
ax.scatter(dataset.x, dataset.y, c=dataset.cluster)


# %%
round(dataset.groupby('cluster').mean(),1)

# %%
