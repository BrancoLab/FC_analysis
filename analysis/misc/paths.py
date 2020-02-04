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
raw_tosort_fld = os.path.join(main_data_fld, 'raw', 'tosort')
raw_tracking_fld = os.path.join(main_data_fld, 'raw', 'tracking')

# --------------------------------- METADATA --------------------------------- #
mice_log = os.path.join(main_dropbox_fld, 'Locomotion_mice.xlsx')
sessions_log = os.path.join(main_dropbox_fld, 'Locomotion_datalog.xlsx')


__all__ = [
    'sessions_log',
    'mice_log',
    'raw_video_fld',
    'raw_metadata_fld',
    'raw_analog_inputs_fld',
    'raw_tosort_fld',
    'raw_tracking_fld',
]