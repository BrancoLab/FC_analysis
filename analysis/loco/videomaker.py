# %%
# Imports
import sys
sys.path.append('./')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from fcutils.video.utils import open_cvwriter


from analysis.loco.utils import fetch_tracking, get_frames_state, fetch_tracking_processed
from analysis.misc.paths import output_fld

# %%
# --------------------------------- Variables -------------------------------- #
experiment = 'Circarena'
subexperiment = 'dreadds_sc_to_grn'
mouse = 'CA828'
injected = 'CNO'



# Vars to decorate video
center = (480, 480)
radius = 400
fps=60

# Prepare background
frame_shape = (960, 960)
background = np.zeros((*frame_shape, 3))

_ = cv2.circle(background, center, radius, (255, 255, 255), -1)

# Text stuff
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,950)
fontScale              = 1
lineType               = 2

# %%
# -------------------------------- Fetch data -------------------------------- #

filt = dict(experiment=experiment, subexperiment=subexperiment, mouse=mouse, injected=injected)

bp_tracking = fetch_tracking(for_bp=True, **filt)
bones_tracking = fetch_tracking(for_bp=False, **filt)

tracking = fetch_tracking_processed(**filt)
tracking = get_frames_state(tracking)
n_frames = len(tracking.x)


# %%
# -------------------------------- Write clip -------------------------------- #
savepath = os.path.join(output_fld, f'{mouse}_{injected}.mp4')
writer = open_cvwriter(
    savepath, w=frame_shape[0], h=frame_shape[1], 
    framerate=fps, format=".mp4", iscolor=True)

for framen in tqdm(range(n_frames)):
    frame = background.copy()

    # Add head tracking
    points = []
    for i, track in bp_tracking.iterrows():
        if track.bp not in ['snout', 'left_ear', 'right_ear']: continue

        x, y = track.x[framen], track.y[framen]
        if np.isnan(x) or np.isnan(y): 
            continue
        points.append(np.array([y, x], np.int32))
    if len(points) == 3:
        points = np.array(points).reshape((-1, 1, 2))
        cv2.fillConvexPoly(frame, points,(0,0,255))

    # Add boxy tracking
    points = []
    for i, track in bp_tracking.iterrows():
        if track.bp not in ['left_ear', 'right_ear', 'body']: continue
        x, y = track.x[framen], track.y[framen]

        if np.isnan(x) or np.isnan(y): 
            continue
        points.append(np.array([y, x], np.int32))

    if len(points) == 3:
        points = np.array(points).reshape((-1, 1, 2))
        cv2.fillConvexPoly(frame, points,(0,255,0))
    
    # Add marker to see if it's locomoting
    if tracking.state[framen] == 'stationary':
        color = (200, 200, 200)
    elif tracking.state[framen] == 'left_turn':
        color = (255, 50, 50)
    elif tracking.state[framen] == 'right_turn':
        color = (50, 255, 50)
    else:
        color = (50, 50, 255)

    cv2.circle(frame, (75, 75), 50, color, -1)
    cv2.putText(frame, f'Status: {tracking.state[framen]}', 
        (10, 950), 
        font, 
        fontScale,
        color,
        lineType)

    writer.write(frame.astype(np.uint8))

writer.release()


# %%
