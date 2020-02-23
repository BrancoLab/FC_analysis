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
        return pd.DataFrame(q.fetch1())
    else:
        return pd.DataFrame(q.fetch())



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
    left_turn_cluster = clusters_means.angular_velocity.idxmax()
    right_turn_cluster = clusters_means.angular_velocity.idxmin()

    # stationary_cluster = 3
    # slow_cluster = 0
    # running_cluster = 1
    # slow_left_turn_cluster = 5
    # left_turn_cluster = 2
    # slow_right_turn_cluster = 6
    # right_turn_cluster = 4


    clusters = dict(stationary = stationary_cluster,
                    running = running_cluster,
                    left_turn = left_turn_cluster,
                    right_turn = right_turn_cluster,
                    )

    clusters_lookup = {v:k for k,v in clusters.items()}

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

        data[mouse] = track
    return data