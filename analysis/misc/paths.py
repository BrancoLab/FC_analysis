import os
import sys

# Define a bunch of paths
if sys.platform != 'darwin':
    main_data_fld = "Z:\\swc\\branco\\Federico\\Locomotion"
    main_dropbox_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion"
else:
    raise NotImplementedError

# --------------------------------- RAW DATA --------------------------------- #

raw_video_fld = os.path.join(main_data_fld, 'raw', 'video')
raw_metadata_fld = os.path.join(main_data_fld, 'raw', 'metadata')
raw_analog_inputs_fld = os.path.join(main_data_fld, 'raw', 'analog_inputs')


# --------------------------------- METADATA --------------------------------- #
mice_log = os.path.join(main_dropbox_fld, 'Locomotion_mice.xls')