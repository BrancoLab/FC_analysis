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

from fcutils.video.utils import open_cvwriter, get_cap_from_file, get_cap_selected_frame
from fcutils.file_io.utils import listdir

from analysis.loco.utils import fetch_tracking, get_frames_state, fetch_tracking_processed, get_when_in_center
from analysis.misc.paths import output_fld, raw_video_fld
from analysis.dbase.tables import Session


# %%
# --------------------------------- Variables -------------------------------- #
experiment = 'Circarena'
subexperiment = 'dreadds_sc_to_grn'
mouse = 'CA826'
injected = 'CNO'
use_real_video = True



# Vars to decorate video
center = (480, 480)
radius = 350
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
tracking = get_when_in_center(tracking, center, radius)
n_frames = len(tracking.x)

if use_real_video:
    data = Session._get_formatted_date(tracking.date.iloc[0])

    video = [f for f in listdir(raw_video_fld) if mouse in f and data in f and f.endswith(".mp4")][0]
    videocap = get_cap_from_file(video)


# %%
# -------------------------------- Write clip -------------------------------- #
savepath = os.path.join(output_fld, f'{mouse}_{injected}_manyclusters.mp4')
writer = open_cvwriter(
    savepath, w=frame_shape[0], h=frame_shape[1], 
    framerate=fps, format=".mp4", iscolor=True)


for framen in tqdm(range(n_frames)):
    # if not tracking.in_center[framen]:
    #     continue
    
    if not tracking.cluster[framen] in [4, 6]:
        continue

    if use_real_video:
        frame = get_cap_selected_frame(videocap, framen)
    else:
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
    colors = dict(stationary = (180, 180, 180),
                    walking = (255, 255, 255),
                    running = (255, 180, 180),
                    left_turn = (180, 255, 180),
                    right_turn = (180, 180, 255),
                    slow_left_turn = (140, 240, 140),
                    slow_right_turn = (140, 140, 240),)


    cv2.circle(frame, (75, 75), 50, colors[tracking.state[framen]], -1)
    cv2.putText(frame, f'Status: {tracking.state[framen]}', 
        (10, 950), 
        font, 
        fontScale,
        colors[tracking.state[framen]],
        lineType)

    writer.write(frame.astype(np.uint8))

    if framen > 2500:
        break
writer.release()


# %%