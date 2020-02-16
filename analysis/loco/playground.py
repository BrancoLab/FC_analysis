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
test = cno.tracking_data.iloc[0]

# %%
centered = False

f, axarr = create_figure(subplots=True, ncols=3, figsize=(27, 9))

for d_n, (dataset, data) in enumerate(datasets.items()):
    if not data.mice: continue

    color = list(data.colors.values())[-1]

    bouts = data.turns

    for n, (i, bout) in enumerate(bouts.iterrows()):
        axarr[d_n].plot(bout.x, bout.y, color=color)
        if n ==20: break

    axarr[d_n].set(title=dataset+f'  {len(bouts)} bouts', xticks=[], yticks=[])

clean_axes(f)

# %%
