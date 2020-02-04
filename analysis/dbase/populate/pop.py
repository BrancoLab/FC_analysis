import sys
import os
sys.path.append(os.getcwd())

from analysis.dbase.tables import *
from analysis.dbase.utils import sort_mantis_files, get_not_converted_videos

run_prelims=True
CONVERT_VIDEOS = False # Set as true to convert video locally, else make bash script for HPC
FPS = 60

# ---------------------------------------------------------------------------- #
#                                    PRELIMS                                   #
# ---------------------------------------------------------------------------- #
if run_prelims:
    # sort_mantis_files()
    to_convert = get_not_converted_videos(CONVERT_VIDEOS, fps=FPS)


# ---------------------------------------------------------------------------- #
#                                   POPULATE                                   #
# ---------------------------------------------------------------------------- #

# Mouse().pop()

# Experiment().pop()
# print(Experiment())

# Subexp().pop()
# Subexp().show()

# Session().pop()
# Session().pop_metadata()
# print("\n\n --- SESSION ---\n")
# print(Session())
# print((Session * Session.Metadata))