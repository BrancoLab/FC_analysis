import os
import logging
from fancylog import fancylog
import fancylog as package
from shutil import copyfile

from behaviour.tdms.mantis_videoframes_test import check_mantis_dropped_frames
from fcutils.file_io.utils import listdir, get_subdirs

from analysis.misc.paths import raw_tosort_fld, raw_video_fld, raw_metadata_fld, raw_analog_inputs_fld


def sort_files():
    # Start logging
    fancylog.start_logging(raw_tosort_fld, package, verbose=verbose)
    logging.info("Starting to process mantis files")
    exp_dirs = get_subdirs(raw_tosort_fld)

    # Loop over subdirs
    for subdir in exp_dirs:
        # --------------------------------- GET FILES -------------------------------- #
        logging.info("  {}".format(subdir))
        files = listdir(subdir)
        if not files: continue
        if len(files) > 3:
            raise NotImplementedError("Can't deal with this many files!")
        elif len(files)<3:
            raise ValueError("Found too few files")

        # Infer what the experiment name is
        metadata_file = [f for f in files if 'meta.tmds' in f]
        if not metadata_file:
            logging.warning("Failed to find metadata file")
            raise FileNotFoundError("Could not find metadata file")
        else:
            metadata_file = metadata_file[0]
        
        # Get AI file
        exp_name = os.path.split(metadata_file)[-1].split("(0)")[0]
        inputs_file = [f for f in files if f.endswith(exp_name+'.tdms')]

        if not inputs_file:
            logging.warning("Failed to find analog inputs file")
            raise FileNotFoundError("Could not find analog inputs file")
        else:
            inputs_file = inputs_file[0]
        
        # Get video file
        video_file = files.pop(metadata_file).pop(inputs_file)[0]

        # ---------------------- CHECK IF MANTIS DROPPED FRAMES ---------------------- #
        camera_name = os.path.split(video_file)[-1].split("(0)-")[-1].split(".")[0]
        check = check_mantis_dropped_frames(SUBDIR, camera_name, exp_name, 
                        skip_analog_inputs=True)
        if check:
            logging.info("      Mantis didn't drop any frames for video file")
        else:
            logging.info("      Mantis dropped some frames, darn it.")

        # -------------------------------- MOVE FILES -------------------------------- #
        # Get destination files
        vdest = os.path.join(raw_video_fld, subdir+'_video.tdms')
        mdest = os.path.join(raw_video_fld, subdir+'_video_metadata.tdms')
        adest = os.path.join(raw_analog_inputs_fld, subdir+'_AI.tdms')

        logging.info("      Video file: {} -> {}".format(video_file, vdest))
        logging.info("      Metadata file: {} -> {}".format(metadata_file, mdest))
        logging.info("      Analog inputs file: {} -> {}".format(inputs_file, adest))

        # Move files 
        copyfile(video_file, vdest)
        copyfile(metadata_file, mdest)
        copyfile(inputs_file, adest)

        logging.info("  moving completed. Checking if everything went okay.")

        # ------------------------------- DELETE FILES ------------------------------- #
        for src, dest in zip([video_file, metadata_file, inputs_file],[vdest, mdest, adest]):
            if os.path.get_size(src) == os.path.get_size(dest):
                log.info("      in theory we would be deleting {}, but let's check this".format(src))
                # os.remove(src)



if __name__ == "__main__":
    sort_files()