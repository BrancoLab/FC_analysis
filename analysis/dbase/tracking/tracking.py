import sys
sys.path.append("./")

try:
    import deeplabcut as dlc
except ModuleNotFoundError:
    print("Could not import deeplabcut")

from fcutils.file_io.utils import listdir
from behaviour.tracking.tracking import prepare_tracking_data

from analysis.misc.paths import dlc_config_file, raw_tracking_fld, processed_tracking_fld
from analysis.dbase.tracking.utils import get_not_tracked_files


# ---------------------------------------------------------------------------- #
#                  TRACK VIDEOS THAT HAVEN'T BEEN TRACKED YET                  #
# ---------------------------------------------------------------------------- #
def track_videos():
    to_track = get_not_tracked_files()

    # Track
    if to_track:
        dlc.analyze_videos(dlc_config_file, to_track, 
                        destfolder=raw_tracking_fld,
                        videotype='.mp4', save_as_csv=False,
                        dynamic=(True, 0.5, 100))

    # Rename files to smth sensible
    # Find file names after filtering. 

    # Move files to processed folder
    # TODO: Rename files to smth sensible



if __name__ == "__main__":
        track_videos()

