import os
import logging
from fancylog import fancylog
import fancylog as package
from tqdm import tqdm

import sys
logging.disable(sys.maxsize)

from behaviour.tdms.mantis_videoframes_test import check_mantis_dropped_frames
from fcutils.file_io.utils import listdir, get_subdirs, get_file_name
from tdmstovideo.converter import convert as tdmsconvert

from analysis.misc.paths import *
from analysis.misc.paths import bash_scripts, hpc_raw_video_fld, hpc_raw_metadata_fld

# ---------------------------------------------------------------------------- #
#                          DATABASE POPULATION HELPERS                         #
# ---------------------------------------------------------------------------- #


# ------------------------------ Videos to track ----------------------------- #
def get_not_tracked_files():
    videos = [f for f in listdir(raw_video_fld) if f.endswith(".mp4")]
    tracked = [get_file_name(f) for f in listdir(raw_tracking_fld) if f.endswith(".h5")]

    not_tracked = []
    for video in videos:
        is_tracked = [f for f in tracked if get_file_name(video) in f]
        if not is_tracked:
            not_tracked.append(video)

    print(f"Found {len(not_tracked)} not tracked videos")
    for vid in not_tracked:
        print(f"     {vid}")
    return not_tracked


# ----------------------------- Videos to convert ---------------------------- #
def get_not_converted_videos(convert_videos, fps=None):
    raw_vids = [f for f in listdir(raw_video_fld) if f.endswith(".tdms")]
    converted = [f.split(".")[0] for f in listdir(raw_video_fld) if f.endswith(".mp4")]

    to_convert = []
    for raw in raw_vids:
        if raw.split(".")[0] not in converted:
            to_convert.append(raw)

    print("\n\nFound {} videos to convert: ".format(len(to_convert))) 
    for vid in to_convert:
        print("     ", vid)     

    
    if convert_videos:
        for video in raw_vids:
            # Get metadata file
            vid = get_file_name(video)
            metadata = [f for f in listdir(raw_metadata_fld) if vid in f][0]
            tdmsconvert(video, metadata, fps=fps)
    else:
        for video in to_convert:
            name = get_file_name(video)
            metadata = [f for f in listdir(raw_metadata_fld) if name in f][0]

            # Create sbatch scripts to run the conversion on HPC
            template = open(os.path.join(bash_scripts, "run_on_hpc_template.txt"), "r").read()

            newbash = template.replace("out.out", "output/{}.out".format(name))
            newbash = newbash.replace("err.err", "output/{}.err".format(name))

            newbash = newbash.replace("VIDEO", os.path.join(hpc_raw_video_fld, os.path.split(video)[-1]).replace("\\", "/"))
            newbash = newbash.replace("METADATA", os.path.join(hpc_raw_metadata_fld, os.path.split(metadata)[-1]).replace("\\", "/"))
            newbash = newbash.replace("FPS",  str(fps))
            
            # Write output
            script_name = os.path.join(bash_scripts, "individuals", name+".sh")
            f = open(script_name,"w")
            f.write(newbash)
            f.close() 
            print("Created bash script at: " + script_name)


# ----------------------------- Sort mantis files ---------------------------- #
def sort_mantis_files():
    exp_dirs = get_subdirs(raw_tosort_fld)

    if not exp_dirs:
        return

    # Start logging
    logging.disable(logging.NOTSET)
    fancylog.start_logging(raw_tosort_fld, package, verbose=True, filename='mantis_sorter')
    logging.info("Starting to process mantis files")

    # Loop over subdirs
    for subdir in exp_dirs:
        # --------------------------------- GET FILES -------------------------------- #
        logging.info("  processing: {}".format(subdir))
        files = [f for f in listdir(subdir) if f.endswith('.tdms')]
        if not files: continue
        if len(files) > 3:
            raise NotImplementedError("Can't deal with this many files!")
        elif len(files)<3:
            raise ValueError("Found too few files")
    
        for f in files:
            for i in range(10):
                if "({})".format(i+1) in f:
                    raise NotImplementedError("Cannot deal with how files are organised in the folder, sorry. ")

        # Infer what the experiment name is
        metadata_file = [f for f in files if 'meta.tdms' in f]
        if not metadata_file:
            logging.warning("Failed to find metadata file")
            raise FileNotFoundError("Could not find metadata file")
        else:
            metadata_file = metadata_file[0]
        
        # Get AI file
        exp_name = os.path.split(metadata_file)[-1].split("(0)")[0]
        inputs_file = [f for f in files if f.endswith(exp_name+'(0).tdms')]

        if not inputs_file:
            logging.warning("Failed to find analog inputs file")
            raise FileNotFoundError("Could not find analog inputs file")
        else:
            inputs_file = inputs_file[0]
        
        # Get video file
        video_file = [f for f in files if f != inputs_file and f != metadata_file][0]

        # ---------------------- CHECK IF MANTIS DROPPED FRAMES ---------------------- #
        camera_name = os.path.split(video_file)[-1].split("(0)-")[-1].split(".")[0]
        check = check_mantis_dropped_frames(subdir, camera_name, exp_name, 
                        skip_analog_inputs=True)
        if check:
            logging.info("      Mantis didn't drop any frames for video file")
        else:
            logging.info("      Mantis dropped some frames, darn it.")

        # -------------------------------- MOVE FILES -------------------------------- #
        # Get destination files
        subshort =  os.path.split(subdir)[-1]
        vdest = os.path.join(raw_video_fld, subshort+'_video.tdms')
        mdest = os.path.join(raw_metadata_fld, subshort+'_video_metadata.tdms')
        adest = os.path.join(raw_analog_inputs_fld, subshort+'_AI.tdms')

        logging.info("      Video file: {} -> {}".format(video_file, vdest))
        logging.info("      Metadata file: {} -> {}".format(metadata_file, mdest))
        logging.info("      Analog inputs file: {} -> {}".format(inputs_file, adest))

        # Move files
        for src, dest in tqdm(zip([video_file, metadata_file, inputs_file],[vdest, mdest, adest])): 
            if os.path.isfile(dest):
                logging.warning("      The destination file {} already exists, stopping to avoid overwriting".format(dest))
                raise FileExistsError("      The destination file {} already exists, stopping to avoid overwriting".format(dest))
            os.rename(src, dest)


    # disable logging
    logging.disable(sys.maxsize)


