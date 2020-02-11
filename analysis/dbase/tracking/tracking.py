import sys
sys.path.append("./")

try:
    import deeplabcut as dlc
except ModuleNotFoundError:
    print("Could not import deeplabcut")

from tqdm import tqdm

from fcutils.file_io.utils import listdir, get_file_name
from fcutils.file_io.io import save_json
from behaviour.tracking.tracking import prepare_tracking_data, compute_body_segments

from analysis.misc.paths import dlc_config_file, raw_tracking_fld, processed_tracking_fld
from analysis.dbase.tracking.utils import get_not_tracked_files
from analysis.dbase.tables import Tracking


TRACK_VIDEOS = True
EXPAND_TRACKINGS = False

# ---------------------------------------------------------------------------- #
#                  TRACK VIDEOS THAT HAVEN'T BEEN TRACKED YET                  #
# ---------------------------------------------------------------------------- #
def track_videos():
    to_track = get_not_tracked_files()

    # Track
    dlc.analyze_videos(dlc_config_file, to_track, 
                    destfolder=raw_tracking_fld,
                    save_as_csv=False,
                    dynamic=(True, 0.5, 100))

    # Rename files to smth sensible
    # Find file names after filtering. 

    # Move files to processed folder
    # TODO: Rename files to smth sensible



# ---------------------------------------------------------------------------- #
#                          CLEAN/EXPAND TRACKING DATA                          #
# ---------------------------------------------------------------------------- #
def expand_tracking_data():
    # Get files not processed yet
    raws = [f for f in list_dir(raw_tracking_fld) if f.endswith('.h5')]
    processed = [f for f in list_dir(processed_tracking_fld) if f.endswith('.json')]
    processed_names = [get_file_name(f) for f in processed]
    to_process = [f for f in raws if get_file_name(f) not in processed_names]
    print(f'Found {len(to_process)} files to process, getting to work.')

    # Clean them up!
    for tracking_file in tqdm(to_process):
        # Process data
        tracking = prepare_tracking_data(tracking_file, likelihood_th=0.999, compute=True)
        bones = compute_body_segments(tracking, Trackings.bsegments)

        # Save
        filename = get_file_name(tracking_file)
        save_json(os.path.join(processed_tracking_fld, filename+'.json'), tracking)
        save_json(os.path.join(processed_tracking_fld, filename+'_bones.json'), bones)


if __name__ == "__main__":
    if TRACK_VIDEOS:
        track_videos()
        
    if EXPAND_TRACKINGS:
        expand_tracking_data()