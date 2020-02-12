import os

from fcutils.file_io.utils import listdir, get_subdirs, get_file_name

from analysis.misc.paths import *

def get_not_tracked_files():
    videos = [f for f in listdir(raw_video_fld) if f.endswith(".mp4")]
    tracked = [get_file_name(f) for f in listdir(raw_tracking_fld) if f.endswith(".h5")]

    not_tracked = []
    for video in videos:
        if get_file_name(video) not in tracked:
            not_tracked.append(video)

    print(f"Found {len(not_tracked)} not tracked videos")
    for vid in not_tracked:
        print(f"     {vid}")
    return not_tracked

