import sys
sys.path.append("./")

try:
    import deeplabcut as dlc
except ModuleNotFoundError:
    print("Could not import deeplabcut")

from tqdm import tqdm
import os

from fcutils.file_io.utils import listdir, get_file_name
from fcutils.file_io.io import save_json
from behaviour.tracking.tracking import prepare_tracking_data, compute_body_segments

from analysis.dbase.tracking.utils import get_not_tracked_files
from analysis.dbase.tables import Tracking
from analysis.misc.paths import bash_scripts, hpc_raw_tracking_fld, hpc_dlc_config_file, hpc_raw_video_fld
from analysis.misc.paths import dlc_config_file, raw_tracking_fld


# ---------------------------------------------------------------------------- #
#                  TRACK VIDEOS THAT HAVEN'T BEEN TRACKED YET                  #
# ---------------------------------------------------------------------------- #
def track_videos(track=False):
    to_track = get_not_tracked_files()

    # Track
    if to_track and track:
        dlc.analyze_videos(dlc_config_file, to_track, 
                        destfolder=raw_tracking_fld,
                        videotype='.mp4', save_as_csv=False,
                        dynamic=(True, 0.5, 100))
    elif to_track:
        # Create bash scripts for HPC
        for video in to_track:
            name = get_file_name(video)
            template = open(os.path.join(bash_scripts, "dlc_on_hpc_template.txt"), "r").read()

            newbash = template.replace("out.out", "output/{}.out".format(name))
            newbash = newbash.replace("err.err", "output/{}.err".format(name))

            newbash = newbash.replace("CONFIG", hpc_dlc_config_file)
            newbash = newbash.replace("VIDEO", os.path.join(hpc_raw_video_fld, os.path.split(video)[-1]).replace("\\", "/"))
            newbash = newbash.replace("DEST", hpc_raw_tracking_fld)

            script_name = os.path.join(bash_scripts, "dlc_individuals", name+".sh")
            f = open(script_name,"w")
            f.write(newbash)
            f.close() 
            print("Created bash script at: " + script_name)


if __name__ == "__main__":
    track_videos()

