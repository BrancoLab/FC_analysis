import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

from fcutils.maths.geometry import calc_distance_from_point
from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d, calc_distance_between_points_2d
from fcutils.plotting.colors import salmon, colorMap, goldenrod, desaturate_color
from fcutils.maths.filtering import median_filter_1d

from behaviour.utilities.signals import get_times_signal_high_and_low

from analysis.dbase.tables import Session, Tracking, ProcessedMouse


# --------------------------------- Fetchers --------------------------------- #
def fetch_tracking_processed(experiment = None, subexperiment = None, 
                        mouse = None, injected = None):
    q = (Session  * Session.IPinjection * ProcessedMouse)

    if experiment is not None:
        q = q & f'exp_name="{experiment}"'

    if subexperiment is not None:
        q = q & f'subname="{subexperiment}"'

    if mouse is not None:
        q = q & f'mouse_id="{mouse}"'
    
    if injected is not None:
        q = q & f'injected="{injected}"'

    if len(q) == 1:
        return pd.DataFrame(q.fetch1())
    else:
        return pd.DataFrame(q.fetch())



def fetch_tracking(for_bp=True,
                        bp = None, bp1 = None, bp2 = None,
                        experiment = None, subexperiment = None, 
                        mouse = None, injected = None):
    q = (Session  * Session.IPinjection * Tracking)

    if for_bp:
        q = q * Tracking.BodyPartTracking

        if bp is not None:
            q = q & f"bp='{bp}'"

    else:
        q = q * Tracking.BodySegmentTracking

        if bp1 is not None:
            q = q & f"1='{bp1}'"

        if bp2 is not None:
            q = q & f"bp='{bp2}'"


    if experiment is not None:
        q = q & f'exp_name="{experiment}"'

    if subexperiment is not None:
        q = q & f'subname="{subexperiment}"'

    if mouse is not None:
        q = q & f'mouse_id="{mouse}"'
    
    if injected is not None:
        q = q & f'injected="{injected}"'

    if len(q) == 1:
        return pd.DataFrame(q.fetch1())
    else:
        return pd.DataFrame(q.fetch())







# --------------------------------- Analysis --------------------------------- #
def get_frames_state(tracking):
    # Standardise
    try:
        speed = preprocessing.scale(np.nan_to_num(tracking.speed.values))
        angular_velocity = preprocessing.scale(np.nan_to_num(tracking.ang_vel.values))
    except Exception as e:
        speed = preprocessing.scale(np.nan_to_num(tracking.speed))
        angular_velocity = preprocessing.scale(np.nan_to_num(tracking.ang_vel))

    # Fit kmeans
    dataset = pd.DataFrame(dict(speed=speed, angular_velocity=angular_velocity))
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    res = kmeans.fit(dataset)

    # Get cluster and state
    y_kmeans = kmeans.fit_predict(dataset)
    tracking['cluster'] = y_kmeans
    dataset['cluster'] = y_kmeans

    # Get state from clusters
    clusters_means = round(dataset.groupby('cluster').mean(),1)

    stationary_cluster = clusters_means.speed.idxmin()
    running_cluster = clusters_means.speed.idxmax()
    left_turn_cluster = clusters_means.angular_velocity.idxmin()
    right_turn_cluster = clusters_means.angular_velocity.idxmax()

    clusters = dict(stationary = stationary_cluster,
                            running = running_cluster,
                            left_turn = left_turn_cluster,
                            right_turn = right_turn_cluster)

    clusters_lookup = {v:k for k,v in clusters.items()}

    state = [clusters_lookup[v] for v in y_kmeans]
    tracking['state'] = state

    return tracking
    




















# # ---------------------------------------------------------------------------- #
# #                                     UTILS                                    #
# # ---------------------------------------------------------------------------- #
# def get_tracking_speed(fps, keep_min, tracking_data, mouse):

#     mouse_tracking = tracking_data.loc[tracking_data.mouse_id == mouse]
#     x , y = mouse_tracking.x.values[0], mouse_tracking.y.values[0]

#     if keep_min is None:
#         keep_frames = len(x)
#     else:
#         keep_frames = fps*60*keep_min

#     XY = np.vstack([x[:keep_frames], y[:keep_frames]])
#     speed = mouse_tracking['speed'].values[0][:keep_frames]
#     ang_vel = mouse_tracking['angular_velocity'].values[0][:keep_frames]
#     dir_of_mvmt = mouse_tracking['dir_of_mvmt'].values[0][:keep_frames]
#     body_orientation = mouse_tracking['body_orientation'].values[0][:keep_frames]
#     body_ang_vel = mouse_tracking['body_ang_vel'].values[0][:keep_frames]

#     return XY, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel


# def get_tracking_in_center(tracking_data, center, radius, fps, keep_min, reverse=False):
#     in_center = tracking_data.copy()

#     for i, row in in_center.iterrows():
#         xy, speed, _, _, _, _ = get_tracking_speed( fps, keep_min, in_center, row.mouse_id)
#         dist = calc_distance_from_point(xy.T, center)

#         if not reverse:
#             to_exclude = np.where(dist > radius)[0]
#         else:
#             to_exclude = np.where(dist < radius)[0]

#         for col in ['x', 'y', 'speed', 'dir_of_mvmt', 'angular_velocity', 'body_orientation', 'body_ang_vel']:
#             vals = row[col].copy()
#             vals[to_exclude] = np.nan
#             row[col] = vals
        
#         in_center.iloc[i] = row
#     return in_center

# def get_tracking_locomoting(tracking_data, center, radius, fps, keep_min, speed_th):
#     locomoting = tracking_data.copy()

#     for i, row in locomoting.iterrows():
#         xy, speed, _, _, _, _ = get_tracking_speed( fps, keep_min, locomoting, row.mouse_id)
#         too_slow = np.where(speed <= speed_th)[0]

#         for col in ['x', 'y', 'speed', 'dir_of_mvmt', 'angular_velocity', 'body_orientation', 'body_ang_vel']:
#             vals = row[col].copy()
#             vals[too_slow] = np.nan
#             row[col] = vals
#         locomoting.iloc[i] = row
#     return locomoting



# # ---------------------------------------------------------------------------- #
# #                  CREATE ARRAY OF INDIVIDUAL LOCOMOTION BOUTS                 #
# # ---------------------------------------------------------------------------- #
# def get_bouts_df(tracking_data, fps, keep_min, min_dist=10, min_dur=20):
#     if 'neck_x' in tracking_data.columns:
#         bouts = dict(start=[], end=[],  x=[], y=[], ang_vel=[], dir_of_mvmt=[], speed=[],
#                         body_orientation=[], body_ang_vel=[], neck_x=[], neck_y=[],
#                         distance=[], torosity=[])

#     else:
#         bouts = dict(start=[], end=[],  x=[], y=[], ang_vel=[], dir_of_mvmt=[], speed=[],
#                         body_orientation=[], body_ang_vel=[],
#                         distance=[], torosity=[])

#     mice = list(set(tracking_data.mouse_id.values))
#     for i, mouse in enumerate(mice):
#         mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
#                                  = get_tracking_speed(fps, keep_min, tracking_data, mouse)

#         x, y = mouse_tracking[0, :], mouse_tracking[1, :]
#         out_of_bounds = np.where(np.isnan(x))[0]
#         indices = np.ones_like(x)
#         indices[out_of_bounds] = 0
#         onsets, offsets = get_times_signal_high_and_low(indices, th=.1)
#         if offsets[0] < onsets[0]:
#             offsets = offsets[1:]

#         if 'neck_x' in tracking_data.columns:
#             mouse_tracking = tracking_data.loc[tracking_data.mouse_id == mouse]
#             nx , ny = mouse_tracking.neck_x.values[0], mouse_tracking.neck_y.values[0]

#         for onset, offset in zip(onsets, offsets):
#             if onset > offset:
#                 raise ValueError
#             if offset - onset < 2: 
#                 continue
#             _x, _y = x[onset+1:offset], y[onset+1:offset]

#             bouts['start'].append(onset+1)
#             bouts['end'].append(offset)
#             bouts['x'].append(_x)
#             bouts['y'].append(_y)
#             bouts['ang_vel'].append(ang_vel[onset+1:offset])
#             bouts['speed'].append(speed[onset+1:offset])
#             bouts['dir_of_mvmt'].append(dir_of_mvmt[onset+1:offset])
#             bouts['body_orientation'].append(body_orientation[onset+1:offset])
#             bouts['body_ang_vel'].append(body_ang_vel[onset+1:offset])


#             if 'neck_x' in tracking_data.columns:
#                 bouts['neck_x'].append(nx[onset+1:offset])
#                 bouts['neck_y'].append(ny[onset+1:offset])


#             distance = np.sum(calc_distance_between_points_in_a_vector_2d(_x, _y))
#             min_dist = calc_distance_between_points_2d((_x[0], _y[0]), (_x[-1], _y[-1]))
            
#             bouts['distance'].append(distance)
#             bouts['torosity'].append(distance/min_dist)


#     bouts = pd.DataFrame(bouts)
#     bouts['duration'] = bouts['end'] - bouts['start']

#     bouts = bouts.loc[(bouts.duration > min_dur)&(bouts.distance > min_dist)]
#     return bouts

# def get_turns_df(tracking_data, tracking_data_in_center, fps, keep_min, min_dist=10, min_dur=20):
#     bouts = dict(start=[], end=[],  x=[], y=[], ang_vel=[], dir_of_mvmt=[], speed=[],
#                     body_orientation=[], body_ang_vel=[],
#                     distance=[], torosity=[])

#     mice = list(set(tracking_data.mouse_id.values))
#     for i, mouse in enumerate(mice):
#         _, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
#                                  = get_tracking_speed(fps, keep_min, tracking_data, mouse)
#         mouse_tracking, speed, _, _, _, _ = get_tracking_speed(fps, keep_min, tracking_data_in_center, mouse)

#         x, y = mouse_tracking[0, :], mouse_tracking[1, :]

#         no_turn = np.where(np.abs(body_ang_vel) < 1000.)[0]
#         out_of_bound = np.where(np.isnan(x))[0]
#         indices = np.ones_like(x)
#         indices[no_turn] = 0
#         indices[out_of_bound] = 0

#         onsets, offsets = get_times_signal_high_and_low(indices, th=.1)
#         if offsets[0] < onsets[0]:
#             offsets = offsets[1:]

#         for onset, offset in zip(onsets, offsets):
#             if onset > offset:
#                 raise ValueError
#             if offset - onset < 2: 
#                 continue
#             _x, _y = x[onset+1:offset], y[onset+1:offset]

#             bouts['start'].append(onset+1)
#             bouts['end'].append(offset)
#             bouts['x'].append(_x)
#             bouts['y'].append(_y)
#             bouts['ang_vel'].append(ang_vel[onset+1:offset])
#             bouts['speed'].append(speed[onset+1:offset])
#             bouts['dir_of_mvmt'].append(dir_of_mvmt[onset+1:offset])
#             bouts['body_orientation'].append(body_orientation[onset+1:offset])
#             bouts['body_ang_vel'].append(body_ang_vel[onset+1:offset])

#             distance = np.sum(calc_distance_between_points_in_a_vector_2d(_x, _y))
#             min_dist = calc_distance_between_points_2d((_x[0], _y[0]), (_x[-1], _y[-1]))
            
#             bouts['distance'].append(distance)
#             bouts['torosity'].append(distance/min_dist)


#     bouts = pd.DataFrame(bouts)
#     bouts['duration'] = bouts['end'] - bouts['start']

#     bouts = bouts.loc[(bouts.duration > min_dur)&(bouts.distance > min_dist)]
#     return bouts



# # ---------------------------------------------------------------------------- #
# #                                FETCH COMPLETE                                #
# # ---------------------------------------------------------------------------- #
# def fetch_entries(entries, bone_entries, cmap, center, radius,  fps, keep_min, speed_th, neck_entries=None):
#     dataset = namedtuple('dataset', 
#             'tracking_data, tracking_data_in_center, tracking_data_not_in_center, \
#             tracking_data_in_center_locomoting, tracking_data_not_in_center_locomoting, \
#             centerbouts, bouts, outbouts, turns,\
#             mice, colors')

#     # Get body trackingdata
#     tracking_data = pd.DataFrame(entries.fetch())

#     # Get neck tracking data if needed
#     if neck_entries is not None:
#         neck_tracking = pd.DataFrame(neck_entries.fetch())
#         tracking_data['neck_x'] = neck_tracking.x.values
#         tracking_data['neck_y'] = neck_tracking.y.values

#     # Get body orientation from bone tracking data
#     bone_tracking_data = pd.DataFrame(bone_entries.fetch())
#     tracking_data['body_orientation'] = bone_tracking_data['orientation']
#     tracking_data['body_ang_vel'] = bone_tracking_data['angular_velocity']

#     # Get the tracking for when the mouse is in the central area and when it is not. 
#     tracking_data_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min)
#     tracking_data_not_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min, reverse=True)

#     # Get the tracking when the mouse is locomoting
#     tracking_data_in_center_locomoting = get_tracking_locomoting(tracking_data_in_center, center, radius,  fps, keep_min, speed_th)
#     tracking_data_not_in_center_locomoting = get_tracking_locomoting(tracking_data_not_in_center, center, radius,  fps, keep_min, speed_th)

#     # Get locomotion bouts
#     centerbouts = get_bouts_df(tracking_data_in_center, fps, keep_min)
#     bouts = get_bouts_df(tracking_data_in_center_locomoting, fps, keep_min)
#     outbouts = get_bouts_df(tracking_data_not_in_center_locomoting, fps, keep_min)

#     # Get turns
#     turns = get_turns_df(tracking_data, tracking_data_in_center, fps, keep_min)

#     # Prepare some more variables
#     mice = list(tracking_data.mouse_id.values)
#     colors = {m:colorMap(i, cmap, vmin=-3, vmax=len(mice)) \
#                         for i,m in enumerate(mice)}

#     return dataset(
#                 tracking_data, 
#                 tracking_data_in_center, 
#                 tracking_data_not_in_center, 
#                 tracking_data_in_center_locomoting,
#                 tracking_data_not_in_center_locomoting,
#                 centerbouts,
#                 bouts,
#                 outbouts,
#                 turns,
#                 mice, 
#                 colors)