import os
import logging
from fancylog import fancylog
import fancylog as package
from tqdm import tqdm

from shutil import copyfile

from behaviour.tdms.mantis_videoframes_test import check_mantis_dropped_frames
from fcutils.file_io.utils import listdir, get_subdirs

from analysis.misc.paths import raw_tosort_fld, raw_video_fld, raw_metadata_fld, raw_analog_inputs_fld


def sort_files():
    # Start logging
    fancylog.start_logging(raw_tosort_fld, package, verbose=True)
    logging.info("Starting to process mantis files")
    exp_dirs = get_subdirs(raw_tosort_fld)

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






if __name__ == "__main__":
    sort_files()