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

from analysis.loco.utils import fetch_tracking, get_frames_state
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
tracking = fetch_tracking(mouse = 'CA826', injected='CNO')



state = get_frames_state(tracking)


# %%
