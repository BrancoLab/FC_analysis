# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fcutils.plotting.utils import create_figure, 
from fcutils.plotting.plotting_elements import plot_shaded_withline
from fcutils.plotting.colors import deepseagreen, colorMap, lilla

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
    return tracking_data, dir_of_mvmt, angular_velocity


# ---------------------------------------------------------------------------- #
#                   Based on direction of movement for mouse                   #
# ---------------------------------------------------------------------------- #

# %%
# ------------------------------- Simple plots ------------------------------- #
# Plot dir of mvmt and angular velocity for each mouse

f, axarr = plt.subplots(ncols=2, nrows=len(mice), sharex=True)

for i, mouse in enumerate(mice):
    tracking_data, dir_of_mvmt, angular_velocity = get_tracking_speed(tracking_data, mouse)

    axarr[0, i].plot(dir_of_mvmt, lw=2, color=lilla)
    axarr[1, i].plot(angular_velocity, lw=2, color=deepseagreen)
    axarr[0, i].set(title=mouse+' dir. of movement')
    axarr[1, i].set(title=mouse+' angular velocity')




# %%
# ---------------------------------------------------------------------------- #
#                                Based on bones                                #
# ---------------------------------------------------------------------------- #





