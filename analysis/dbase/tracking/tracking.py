import deeplabcut as dlc

from fcutils.file_io.utils import listdir
from behaviour.tracking.tracking import prepare_tracking_data

from analysis.misc.paths import dlc_config_file, raw_tracking_fld, processed_tracking_fld
from analysis.dbase.tracking.utils import get_not_tracked_files


TRACK_VIDEOS = True
EXPAND_TRACKINGS = True

# ---------------------------------------------------------------------------- #
#                  TRACK VIDEOS THAT HAVEN'T BEEN TRACKED YET                  #
# ---------------------------------------------------------------------------- #
def track_videos():
    to_track = get_not_tracked_files()

    # Track
    dlc.analyze_videos(dlc_config_file, to_track, 
                    destfolder=raw_tracking_fld,
                    videotype='.mp4', save_as_csv=False,
                    dynamic=(True, 0.5, 100))

    # Median filter 
    deeplabcut.filterpredictions(dlc_config_file, to_track, 
        videotype='mp4', filtertype='median',
        save_as_csv=False, destfolder=raw_tracking_fld)

    # Rename files to smth sensible
    # Find file names after filtering. 

    # Move files to processed folder
    # TODO: Rename files to smth sensible



# ---------------------------------------------------------------------------- #
#                          CLEAN/EXPAND TRACKING DATA                          #
# ---------------------------------------------------------------------------- #
def expand_tracking_data():
    to_process = [f for f in list_dir(processed_tracking_fld) if '_processed' not in f]

    # Clean them up!
    for tracking_file in to_process:
        tracking = pd.read_hdf(tracking_file, key='hdf')
        tracking = prepare_tracking_data(tracking, likelihood_th=0.999, compute=True)

        tracking.to_hdf(tracking_file, key='hdf')


if __name__ == "__main__":
    if TRACK_VIDEOS:
        track_videos()
        
    if EXPAND_TRACKINGS:
        expand_tracking_data()