import numpy as np
import pandas as pd
from collections import namedtuple

from fcutils.maths.geometry import calc_distance_from_point
from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d, calc_distance_between_points_2d
from fcutils.plotting.colors import salmon, colorMap, goldenrod, desaturate_color

from behaviour.utilities.signals import get_times_signal_high_and_low

# ---------------------------------------------------------------------------- #
#                                     UTILS                                    #
# ---------------------------------------------------------------------------- #
def get_tracking_speed(fps, keep_min, tracking_data, mouse):

    mouse_tracking = tracking_data.loc[tracking_data.mouse_id == mouse]
    x , y = mouse_tracking.x.values[0], mouse_tracking.y.values[0]

    if keep_min is None:
        keep_frames = len(x)
    else:
        keep_frames = fps*60*keep_min

    XY = np.vstack([x[:keep_frames], y[:keep_frames]])
    speed = mouse_tracking['speed'].values[0][:keep_frames]
    ang_vel = mouse_tracking['angular_velocity'].values[0][:keep_frames]
    dir_of_mvmt = mouse_tracking['dir_of_mvmt'].values[0][:keep_frames]
    body_orientation = mouse_tracking['body_orientation'].values[0][:keep_frames]
    body_ang_vel = mouse_tracking['body_ang_vel'].values[0][:keep_frames]

    return XY, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel


def get_tracking_in_center(tracking_data, center, radius, fps, keep_min, reverse=False):
    in_center = tracking_data.copy()

    for i, row in in_center.iterrows():
        xy, speed, _, _, _, _ = get_tracking_speed( fps, keep_min, in_center, row.mouse_id)
        dist = calc_distance_from_point(xy.T, center)

        if not reverse:
            to_exclude = np.where(dist > radius)[0]
        else:
            to_exclude = np.where(dist < radius)[0]

        for col in ['x', 'y', 'speed', 'dir_of_mvmt', 'angular_velocity', 'body_orientation', 'body_ang_vel']:
            vals = row[col].copy()
            vals[to_exclude] = np.nan
            row[col] = vals
        
        in_center.iloc[i] = row
    return in_center

def get_tracking_locomoting(tracking_data, center, radius, fps, keep_min, speed_th):
    locomoting = tracking_data.copy()

    for i, row in locomoting.iterrows():
        xy, speed, _, _, _, _ = get_tracking_speed( fps, keep_min, locomoting, row.mouse_id)
        too_slow = np.where(speed <= speed_th)[0]

        for col in ['x', 'y', 'speed', 'dir_of_mvmt', 'angular_velocity', 'body_orientation', 'body_ang_vel']:
            vals = row[col].copy()
            vals[too_slow] = np.nan
            row[col] = vals
        locomoting.iloc[i] = row
    return locomoting



# ---------------------------------------------------------------------------- #
#                  CREATE ARRAY OF INDIVIDUAL LOCOMOTION BOUTS                 #
# ---------------------------------------------------------------------------- #
def get_bouts_df(tracking_data, fps, keep_min, min_dist=10, min_dur=20):
    if 'neck_x' in tracking_data.columns:
        bouts = dict(start=[], end=[],  x=[], y=[], ang_vel=[], dir_of_mvmt=[], speed=[],
                        body_orientation=[], body_ang_vel=[], neck_x=[], neck_y=[],
                        distance=[], torosity=[])

    else:
        bouts = dict(start=[], end=[],  x=[], y=[], ang_vel=[], dir_of_mvmt=[], speed=[],
                        body_orientation=[], body_ang_vel=[],
                        distance=[], torosity=[])

    mice = list(set(tracking_data.mouse_id.values))
    for i, mouse in enumerate(mice):
        mouse_tracking, speed, ang_vel, dir_of_mvmt, body_orientation, body_ang_vel \
                                 = get_tracking_speed(fps, keep_min, tracking_data, mouse)

        x, y = mouse_tracking[0, :], mouse_tracking[1, :]
        out_of_bounds = np.where(np.isnan(x))[0]
        indices = np.ones_like(x)
        indices[out_of_bounds] = 0
        onsets, offsets = get_times_signal_high_and_low(indices, th=.1)
        if offsets[0] < onsets[0]:
            offsets = offsets[1:]

        if 'neck_x' in tracking_data.columns:
            mouse_tracking = tracking_data.loc[tracking_data.mouse_id == mouse]
            nx , ny = mouse_tracking.neck_x.values[0], mouse_tracking.neck_y.values[0]

        for onset, offset in zip(onsets, offsets):
            if onset > offset:
                raise ValueError
            if offset - onset < 2: 
                continue
            _x, _y = x[onset+1:offset], y[onset+1:offset]

            bouts['start'].append(onset+1)
            bouts['end'].append(offset)
            bouts['x'].append(_x)
            bouts['y'].append(_y)
            bouts['ang_vel'].append(ang_vel[onset+1:offset])
            bouts['speed'].append(speed[onset+1:offset])
            bouts['dir_of_mvmt'].append(dir_of_mvmt[onset+1:offset])
            bouts['body_orientation'].append(body_orientation[onset+1:offset])
            bouts['body_ang_vel'].append(body_ang_vel[onset+1:offset])


            if 'neck_x' in tracking_data.columns:
                bouts['neck_x'].append(nx[onset+1:offset])
                bouts['neck_y'].append(ny[onset+1:offset])


            distance = np.sum(calc_distance_between_points_in_a_vector_2d(_x, _y))
            min_dist = calc_distance_between_points_2d((_x[0], _y[0]), (_x[-1], _y[-1]))
            
            bouts['distance'].append(distance)
            bouts['torosity'].append(distance/min_dist)


    bouts = pd.DataFrame(bouts)
    bouts['duration'] = bouts['end'] - bouts['start']

    bouts = bouts.loc[(bouts.duration > min_dur)&(bouts.distance > min_dist)]
    return bouts




# ---------------------------------------------------------------------------- #
#                                FETCH COMPLETE                                #
# ---------------------------------------------------------------------------- #
def fetch_entries(entries, bone_entries, cmap, center, radius,  fps, keep_min, speed_th, neck_entries=None):
    dataset = namedtuple('dataset', 
            'tracking_data, tracking_data_in_center, tracking_data_not_in_center, \
            tracking_data_in_center_locomoting, tracking_data_not_in_center_locomoting, \
            centerbouts, bouts, outbouts, \
            mice, colors')

    # Get body trackingdata
    tracking_data = pd.DataFrame(entries.fetch())

    # Get neck tracking data if needed
    if neck_entries is not None:
        neck_tracking = pd.DataFrame(neck_entries.fetch())
        tracking_data['neck_x'] = neck_tracking.x.values
        tracking_data['neck_y'] = neck_tracking.y.values

    # Get body orientation from bone tracking data
    bone_tracking_data = pd.DataFrame(bone_entries.fetch())
    tracking_data['body_orientation'] = bone_tracking_data['orientation']
    tracking_data['body_ang_vel'] = bone_tracking_data['angular_velocity']

    # Get the tracking for when the mouse is in the central area and when it is not. 
    tracking_data_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min)
    tracking_data_not_in_center = get_tracking_in_center(tracking_data, center, radius,  fps, keep_min, reverse=True)

    # Get the tracking when the mouse is locomoting
    tracking_data_in_center_locomoting = get_tracking_locomoting(tracking_data_in_center, center, radius,  fps, keep_min, speed_th)
    tracking_data_not_in_center_locomoting = get_tracking_locomoting(tracking_data_not_in_center, center, radius,  fps, keep_min, speed_th)

    # Get locomotion bouts
    centerbouts = get_bouts_df(tracking_data_in_center, fps, keep_min)
    bouts = get_bouts_df(tracking_data_in_center_locomoting, fps, keep_min)
    outbouts = get_bouts_df(tracking_data_not_in_center_locomoting, fps, keep_min)

    # Prepare some more variables
    mice = list(tracking_data.mouse_id.values)
    colors = {m:colorMap(i, cmap, vmin=-3, vmax=len(mice)) \
                        for i,m in enumerate(mice)}

    return dataset(
                tracking_data, 
                tracking_data_in_center, 
                tracking_data_not_in_center, 
                tracking_data_in_center_locomoting,
                tracking_data_not_in_center_locomoting,
                centerbouts,
                bouts,
                outbouts,
                mice, 
                colors)