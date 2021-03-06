import os
import sys

# Define a bunch of paths
if sys.platform != 'darwin':
    main_data_fld = "Z:\\swc\\branco\\Federico\\Locomotion"
    main_dropbox_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion"
else:
    main_data_fld = "/Volumes/swc/branco/Federico/Locomotion"
    main_dropbox_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion"

# --------------------------------- RAW DATA --------------------------------- #
raw_video_fld = os.path.join(main_data_fld, 'raw', 'video')
raw_metadata_fld = os.path.join(main_data_fld, 'raw', 'metadata')
raw_analog_inputs_fld = os.path.join(main_data_fld, 'raw', 'analog_inputs')
raw_tosort_fld = os.path.join(main_data_fld, 'raw', 'tosort')
raw_tracking_fld = os.path.join(main_data_fld, 'raw', 'tracking')

# ------------------------------ Processed Data ------------------------------ #
processed_stimuli_fld = os.path.join(main_data_fld, 'processed', 'stimuli')
output_fld = os.path.join(main_dropbox_fld, 'output')
input_fld = os.path.join(main_dropbox_fld, 'input')


# --------------------------------- METADATA --------------------------------- #
mice_log = os.path.join(main_dropbox_fld, 'Locomotion_mice.xlsx')
sessions_log = os.path.join(main_dropbox_fld, 'Locomotion_datalog.xlsx')
experiments_file = os.path.join(main_dropbox_fld, 'experiments.yml')
surgeries_file = os.path.join(main_dropbox_fld, 'surgeries.yml')

# ----------------------------------- MISC ----------------------------------- #
bash_scripts = os.path.join(main_data_fld, 'bash_scripts')

hpc_loco_fld = "/nfs/winstor/branco/Federico/Locomotion"
hpc_raw_video_fld = os.path.join(hpc_loco_fld, 'raw', 'video').replace("\\", "/")
hpc_raw_metadata_fld = os.path.join(hpc_loco_fld, 'raw', 'metadata').replace("\\", "/")
hpc_raw_analog_inputs_fld = os.path.join(hpc_loco_fld, 'raw', 'analog_inputs').replace("\\", "/")
hpc_raw_tosort_fld = os.path.join(hpc_loco_fld, 'raw', 'tosort').replace("\\", "/")
hpc_raw_tracking_fld = os.path.join(hpc_loco_fld, 'raw', 'tracking').replace("\\", "/")

# -------------------------------- DEEPLABCUT -------------------------------- #
dlc_config_file = os.path.join(main_dropbox_fld, 'dlc', 'locomotion-Federico', 'config.yaml')
hpc_dlc_config_file = os.path.join(hpc_loco_fld, 'dlc', 'locomotion-Federico', 'config.yaml').replace("\\", "/")


# ---------------------------------------------------------------------------- #
#                                    ANATOMY                                   #
# ---------------------------------------------------------------------------- #

# --------------------------------- Brainsaw --------------------------------- #
if sys.platform != 'darwin':
    brainsaw_folder = 'Z:\\swc\\branco\\BrainSaw\\'
else:
    brainsaw_folder = '/Volumes/swc/branco/BrainSaw'
cellfinder_cells_folder = os.path.join(brainsaw_folder, 'cellfinder_cells')
injections_folder = os.path.join(brainsaw_folder, 'injections')

cellfinder_out_dir = os.path.join(main_dropbox_fld, 'output/anatomy')







__all__ = [
    'sessions_log',
    'mice_log',
    'raw_video_fld',
    'raw_metadata_fld',
    'raw_analog_inputs_fld',
    'raw_tosort_fld',
    'raw_tracking_fld',
]