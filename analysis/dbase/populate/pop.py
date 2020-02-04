import sys
import os
sys.path.append(os.getcwd())

from analysis.dbase.tables import *
from analysis.dbase.utils import sort_mantis_files, get_not_converted_videos

CONVERT_VIDEOS = True
FPS = 60
# ---------------------------------------------------------------------------- #
#                                    PRELIMS                                   #
# ---------------------------------------------------------------------------- #
sort_mantis_files()
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
# print(Session())