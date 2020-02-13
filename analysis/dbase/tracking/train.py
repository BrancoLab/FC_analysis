import deeplabcut as dlc
import os

from fcutils.file_io.utils import listdir
# from fcutils.video.utils import trim_clip

config_file = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\dlc\\locomotion-Federico\\config.yaml'


dlc.train_network(config_file)

# fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Locomotion\\dlc'
# vids = [os.path.join(fld, '200203_CA8493_video_trim.mp4'), os.path.join(fld, '200204_CA8491_video_trim.mp4'), os.path.join(fld, '200204_CA8494_video_trim.mp4')]
# dlc.extract_outlier_frames(config_file, vids, epsilon=40)

# dlc.merge_datasets(config_file)

# vids = [f for f in listdir(fld) if f.endswith('.mp4')]

# for vid in vids:
#     savepath = vid.split(".")[0]+'_trim.mp4'
#     trim_clip(vid, savepath, start=0.25, stop=0.35)