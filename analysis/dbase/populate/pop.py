import sys
import os
sys.path.append(os.getcwd())

from analysis.dbase.tables import *
from analysis.dbase.utils.utils import sort_mantis_files, get_not_converted_videos, convert_intputs_ais_to_pandas
from analysis.dbase.tracking.tracking import track_videos

run_prelims = False
CONVERT_TDMS_AI = False
CONVERT_VIDEOS = False # Set as true to convert video locally, else make bash script for HPC
TRACK_VIDEOS = False

POPULATE = False
SUMMARY = True
       
FPS = 60

# Session.drop()

if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    #                                    PRELIMS                                   #
    # ---------------------------------------------------------------------------- #
    if run_prelims:
        sort_mantis_files()
        to_convert = get_not_converted_videos(CONVERT_VIDEOS, fps=FPS)

        track_videos(track=TRACK_VIDEOS)

        if CONVERT_TDMS_AI:
            convert_intputs_ais_to_pandas()
        print("\n\n\n")


    # ---------------------------------------------------------------------------- #
    #                                   POPULATE                                   #
    # ---------------------------------------------------------------------------- #
    if POPULATE:
        # ? Mouse
        Mouse().pop()

        # ? Experiment
        Experiment().pop()
        Subexp().pop()

        # ?  Session
        Session().pop()

        # ?  Stimuliing
        # Stimuli().populate(display_progress=True)

        # ? Tracking
        Tracking.populate(display_progress=True)

        # ? Processed tracking
        ProcessedMouse.populate(display_progress=True)


    # ---------------------------------------------------------------------------- #
    #                               PRINT DBASE STATE                              #
    # ---------------------------------------------------------------------------- #
    if SUMMARY:
        print("\n\n\n----------------------------------------------------------------------------")
        print("--- MOUSE ---\n")
        print(Mouse())
        print("\n\n Injections ")
        print((Mouse * Mouse.Injection))

        print("\n\n\n----------------------------------------------------------------------------")
        print("--- EXPERIMENT ---\n")
        Subexp().show()

        print("\n\n\n----------------------------------------------------------------------------")
        print("--- SESSION ---\n")
        print("Session metadata")
        print((Session * Session.Metadata))
        print("\n\n")
        print("Session IP injection data")
        print((Session * Session.IPinjection))

        print("\n\n\n----------------------------------------------------------------------------")
        print("--- TRACKING ---\n")
        print((Tracking * Tracking.BodyPartTracking & "bp='body'"))

        print("\n\n\n----------------------------------------------------------------------------")
        print("--- PROCESSED TRACKING ---\n")
        print((ProcessedMouse()))