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

"""
    Bunch of utils to facilitated the loading and processing of locomotion data
"""


# --------------------------------- Fetchers --------------------------------- #
def fetch_tracking_processed(experiment = None, subexperiment = None, 
                        mouse = None, injected = None, just_mouse=False):
    if not just_mouse:
        q = (Session * ProcessedMouse)
    else:
        q = (Session)

    if experiment is not None:
        q = q & f'exp_name="{experiment}"'

    if subexperiment is not None:
        q = q & f'subname="{subexperiment}"'

    if mouse is not None:
        q = q & f'mouse_id="{mouse}"'
    
    if injected is not None:
        q = q * Session.IPinjection & f'injected="{injected}"'
    if len(q) == 1:
        df =  pd.DataFrame(q.fetch1())
    else:
        df =  pd.DataFrame(q.fetch())

    if df.empty:
        raise ValueError("Could not fetch tracking data processed...")
    else:
        return df



def fetch_tracking(for_bp=True,
                        bp = None, bp1 = None, bp2 = None,
                        experiment = None, subexperiment = None, 
                        mouse = None, injected = None):
    q = (Session * Tracking)

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
        q = q * Session.IPinjection & f'injected="{injected}"'

    if len(q) == 1:
        return pd.DataFrame(q.fetch1())
    else:
        return pd.DataFrame(q.fetch())


# --------------------------------- Analysis --------------------------------- #
def get_frames_state(tracking):
    if not len(tracking):
        return tracking

    # Standardise
    try:
        speed = preprocessing.scale(np.nan_to_num(tracking.speed.values))
        angular_velocity = preprocessing.scale(np.nan_to_num(tracking.ang_vel.values))
    except Exception as e:
        speed = preprocessing.scale(np.nan_to_num(tracking.speed.values[0]))
        angular_velocity = preprocessing.scale(np.nan_to_num(tracking.ang_vel.values[0]))

    # Fit kmeans
    dataset = pd.DataFrame(dict(speed=speed, angular_velocity=angular_velocity))
    kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 42)
    res = kmeans.fit(dataset)

    # Get cluster and state
    y_kmeans = kmeans.fit_predict(dataset)
    tracking['cluster'] = y_kmeans
    dataset['cluster'] = y_kmeans

    # Get state from clusters
    clusters_means = round(dataset.groupby('cluster').mean(),1)

    right_clusters = clusters_means.loc[clusters_means.angular_velocity < -.5]
    right_clusters_index = list(right_clusters.index.values)
    left_clusters = clusters_means.loc[clusters_means.angular_velocity > .5]
    left_clusters_index = list(left_clusters.index.values)

    clusters = dict(
                    left_turn = 'left',
                    right_turn = 'right',
                    )

    running_clusters = clusters_means.loc[
                    (clusters_means.angular_velocity > -.5) &
                    (clusters_means.angular_velocity < .5)].sort_values('speed')
    for i, idx in enumerate(running_clusters.index.values):
        clusters[f'locomotion_{i}'] = idx

    clusters_lookup = {v:k for k,v in clusters.items()}


    # Clean up states
    y_kmeans = ['left' if k in left_clusters_index else 
                'right' if k in right_clusters_index else k
                        for k in y_kmeans]
    state = [clusters_lookup[v] for v in y_kmeans]
    tracking['state'] = state

    return tracking
    
def get_when_in_center(tracking, center, radius):
    xy = np.vstack([tracking.x, tracking.y])

    dist = calc_distance_from_point(xy.T, center)
    to_exclude = np.where(dist > radius)[0]

    in_center = np.ones_like(tracking.x)
    in_center[to_exclude] = 0
    tracking['in_center'] = in_center
    return tracking




def get_center_bouts(datasets):
    all_bouts = {}
    for dn, (dataset, datas) in enumerate(datasets.items()):
        bts = dict(start=[], end=[], speed=[], orientation=[], 
                                    ang_vel=[], x=[], y=[],
                                    duration=[], distance=[],
                                    state=[], in_center=[], mouse=[],
                                    abs_ang_displ=[], ang_displ=[])
        for mouse,data in datas.items():
            in_center = data.in_center.values

            onsets, offsets = get_times_signal_high_and_low(in_center, th=.1)
            if offsets[0] < onsets[0]:
                offsets = offsets[1:]



            # Loop over bouts
            for onset, offset in zip(onsets, offsets):
                onset += 1
                if offset < onset: raise ValueError
                elif offset - onset < 5: continue # skip bouts that are too short
                
                bts['start'].append(onset)
                bts['end'].append(offset)
                bts['duration'].append(offset - onset)
                bts['speed'].append(data.speed.values[onset:offset])
                bts['distance'].append(np.sum(data.speed.values[onset:offset]))
                bts['orientation'].append(data.orientation.values[onset:offset])
                bts['ang_vel'].append(data.ang_vel.values[onset:offset])
                bts['abs_ang_displ'].append(np.sum(np.abs(data.ang_vel.values[onset:offset])))
                bts['ang_displ'].append(np.sum(data.ang_vel.values[onset:offset]))
                bts['x'].append(data.x.values[onset:offset])
                bts['y'].append(data.y.values[onset:offset])
                bts['state'].append(data.state.values[onset:offset])
                bts['in_center'].append(data.in_center.values[onset:offset])
                bts['mouse'].append(mouse)
        all_bouts[dataset] = pd.DataFrame(bts)
    return all_bouts


def get_bouts(datasets, bouts_types=None, only_in_center=True):
    # Expects a dictionary of dataframes with datasets of tracking for each mouse
    if bouts_types is None:
        bouts_types = ['stationary', 'slow', 'running', 'left_turn', 'right_turn']
    
    # Loop over each dataset
    all_bouts = {}
    for dn, (dataset, datas) in enumerate(datasets.items()):
        bouts = {k:[] for k in bouts_types}

        # Loop over each mouse in dataset
        for mouse, data in datas.items():
            tot_frames = len(data)
            state = data.state

            for bout_type in bouts_types:
                # Get a vector which is 1 only when mouse is in cluster state
                is_state = np.zeros(tot_frames)
                if bout_type == 'stationary':
                    states = ['locomotion_0']
                elif bout_type == 'running':
                    states = ['locomotion_3']
                elif bout_type == 'slow':
                    states = ['locomotion_1', 'locomotion_2']
                else:
                    states = [bout_type]
                is_state[state.isin(states)] = 1

                if only_in_center:
                    is_state[data.in_center == 0] = 0

                # Get onsets and offsets of state
                onsets, offsets = get_times_signal_high_and_low(is_state, th=.1)
                if offsets[0] < onsets[0]:
                    offsets = offsets[1:]

                # Prepr a dict to store bouts data
                bts = dict(start=[], end=[], speed=[], orientation=[], 
                            ang_vel=[], x=[], y=[],
                            duration=[], distance=[],
                            state=[], in_center=[], mouse=[],
                            abs_ang_displ=[], ang_displ=[])

                # Loop over bouts
                for onset, offset in zip(onsets, offsets):
                    onset += 1
                    if offset < onset: raise ValueError
                    elif offset - onset < 5: continue # skip bouts that are too short
                    
                    bts['start'].append(onset)
                    bts['end'].append(offset)
                    bts['duration'].append(offset - onset)
                    bts['speed'].append(data.speed.values[onset:offset])
                    bts['distance'].append(np.sum(data.speed.values[onset:offset]))
                    bts['orientation'].append(data.orientation.values[onset:offset])
                    bts['ang_vel'].append(data.ang_vel.values[onset:offset])
                    bts['abs_ang_displ'].append(np.sum(np.abs(data.ang_vel.values[onset:offset])))
                    bts['ang_displ'].append(np.sum(data.ang_vel.values[onset:offset]))
                    bts['x'].append(data.x.values[onset:offset])
                    bts['y'].append(data.y.values[onset:offset])
                    bts['state'].append(data.state.values[onset:offset])
                    bts['in_center'].append(data.in_center.values[onset:offset])
                    bts['mouse'].append(mouse)
                
                bouts[bout_type].append(pd.DataFrame(bts))
        
        # Concatenate the bouts of each type for each mouse?
        all_bouts[dataset] = {k:pd.concat(b) for k,b in bouts.items()}
    return all_bouts



# ---------------------------------------------------------------------------- #
#                                   ALL FETCH                                  #
# ---------------------------------------------------------------------------- #

def get_experiment_data(experiment=None, subexperiment=None, mouse=None, injected=None,
                            center=None, radius=None, only_in_center= None,
                            keep_min=None, fps=60):
    filt = dict(experiment=experiment, subexperiment=subexperiment, injected=injected, mouse=mouse)
    mice = list(set(fetch_tracking_processed(**filt, just_mouse=True).mouse_id.values))
    data = {}
    for i, mouse in enumerate(mice):
        print(f'    processing {mouse} - {i+1} of {len(mice)}')
        tracking = fetch_tracking_processed(experiment=experiment, subexperiment=subexperiment, 
                                                injected=injected, just_mouse=False, mouse=mouse)

        if keep_min is not None:
            keep = keep_min * 60 * fps
            tracking = tracking[:keep]

        track = get_frames_state(tracking.loc[tracking.mouse_id == mouse])
        track = get_when_in_center(track, center, radius)

        if only_in_center is not None:
            track = track.loc[track.in_center == only_in_center]

        if len(track) < 50000: # there's very few frames
            raise ValueError("Something went wrong while fetching data")

        data[mouse] = track
    return data