import sys
import os
sys.path.append(os.getcwd())

from analysis.dbase.tables import *
from analysis.dbase.utils.utils import sort_mantis_files, get_not_converted_videos
from analysis.dbase.tracking.utils import get_not_tracked_files
from analysis.dbase.tracking.tracking import track_videos, expand_tracking_data

run_prelims=True
CONVERT_VIDEOS = False # Set as true to convert video locally, else make bash script for HPC
TRACK_VIDEOS=False

FPS = 60


# ---------------------------------------------------------------------------- #
#                                    PRELIMS                                   #
# ---------------------------------------------------------------------------- #
if run_prelims:
    sort_mantis_files()
    to_convert = get_not_converted_videos(CONVERT_VIDEOS, fps=FPS)
    to_track = get_not_tracked_files()

    if TRACK_VIDEOS:
        track_videos()
        expand_tracking_data()

    print("\n\n\n")


# ---------------------------------------------------------------------------- #
#                                   POPULATE                                   #
# ---------------------------------------------------------------------------- #
# ? Mouse
Mouse().pop()

# ? Experiment
Experiment().pop()
Subexp().pop()

# ?  Session
Session().pop()

# ? Tracking
# Tracking.populate()


# ---------------------------------------------------------------------------- #
#                               PRINT DBASE STATE                              #
# ---------------------------------------------------------------------------- #
print("\n\n --- MOUSE ---\n")
print(Mouse())

print("\n\n --- EXPERIMENT ---\n")
print(Experiment())
print("\n\n")
Subexp().show()


print("\n\n --- SESSION ---\n")
print(Session())
print("\n\n")
print("Session metadata")
print((Session * Session.Metadata))
print("\n\n")
print("Session IP injection data")
print((Session * Session.IPinjection))


print("\n\n --- TRACKING ---\n")
print((Tracking * Tracking.BodyPartTracking & "bp='body'"))
