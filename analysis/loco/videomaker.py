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
from functools import partial
import cv2
from tqdm import tqdm

from fcutils.video.utils import open_cvwriter


from behaviour.utilities.signals import get_times_signal_high_and_low
from analysis.dbase.tables import Session, Tracking
from analysis.loco.utils import get_tracking_speed, fetch_entries
from analysis.misc.paths import output_fld

# %%
# --------------------------------- Variables -------------------------------- #
experiment = 'Circarena'
subexperiment = 'dreadds_sc_to_grn'
mouse = 'CA828'
injected = 'CNO'

speed_th = 1
fps=60
out_fps = fps
keep_min = 60

get_tracking_speed = partial(get_tracking_speed, fps, keep_min)

# Vars to exlude when mice are on the walls
center = (480, 480)
radius = 400

# Prepare background
frame_shape = (960, 960)
background = np.zeros((*frame_shape, 3))

_ = cv2.circle(background, center, radius, (255, 255, 255), -1)


# %%
# -------------------------------- Fetch data -------------------------------- #
print("Fetching bouts")
body_entries = Session  * Session.IPinjection  * Tracking.BodyPartTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp='body'" \
                & f"mouse_id='{mouse}'" & f"injected='{injected}'"
                
bone_entries = Session  * Session.IPinjection  * Tracking.BodySegmentTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp1='neck'" & f"bp2='body'" \
                & f"mouse_id='{mouse}'" & f"injected='{injected}'"
data = fetch_entries(body_entries, bone_entries, 'Greens', center, radius,  
                    fps, keep_min, speed_th)

print("Fetching all tracking")
tracking = {}
for bp in ['snout', 'left_ear', 'right_ear', 'neck', 'body', 'tail']:
    entries = Session  * Session.IPinjection  * Tracking.BodyPartTracking & f"exp_name='{experiment}'" \
                & f"subname='{subexperiment}'" & f"bp='{bp}'" \
                & f"mouse_id='{mouse}'" &  f"injected='{injected}'"
    tracking[bp] = pd.DataFrame(entries.fetch()).iloc[[0]]

print("Prepping some variables")
# Get locomotion times
n_frames = len(tracking['body'].iloc[0].x)
is_locomoting = np.zeros(n_frames)
for i, bout in data.bouts.iterrows():
    is_locomoting[bout.start:bout.end] = 1

# Text stuff
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,950)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


# %%
# -------------------------------- Write clip -------------------------------- #
savepath = os.path.join(output_fld, f'{mouse}_{injected}_turns.mp4')
writer = open_cvwriter(
    savepath, w=frame_shape[0], h=frame_shape[1], 
    framerate=out_fps, format=".mp4", iscolor=True)

for boutn, (i, bout) in tqdm(enumerate(data.turns.iterrows())):
    for framen in range(bout.start, bout.end):
        frame = background.copy()

        # Add text
        cv2.putText(frame, f'Bout: {boutn} of {len(data.turns)}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # Add head tracking
        points = []
        for bp, tr in tracking.items():
            if bp not in ['snout', 'left_ear', 'right_ear']: continue

            x, y = tr.x.values[0][framen], tr.y.values[0][framen]
            if np.isnan(x) or np.isnan(y): 
                continue
            points.append(np.array([y, x], np.int32))
        if len(points) == 3:
            points = np.array(points).reshape((-1, 1, 2))
            cv2.fillConvexPoly(frame, points,(0,0,255))

        # Add boxy tracking
        points = []
        for bp, tr in tracking.items():
            if bp not in ['left_ear', 'right_ear', 'body']: continue
            x, y = tr.x.values[0][framen], tr.y.values[0][framen]

            if np.isnan(x) or np.isnan(y): 
                continue
            points.append(np.array([y, x], np.int32))

        if len(points) == 3:
            points = np.array(points).reshape((-1, 1, 2))
            cv2.fillConvexPoly(frame, points,(0,255,0))
        
        # Add marker to see if it's locomoting
        if not is_locomoting[framen]:
            cv2.circle(frame, (75, 75), 50, (200, 200, 200), -1)
        else:
            cv2.circle(frame, (75, 75), 50, (50, 255, 50), -1)
        writer.write(frame.astype(np.uint8))
writer.release()


# %%
