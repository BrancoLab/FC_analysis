from Utils.imports import *
from copy import deepcopy as cp


def crop_trial_tracking(trial):
    # TODO add fps extraction
    midpoint = 1800
    range_to_keep_s = [10, 20]  # n seconds before and after the midpoin to keep
    range_to_keep = tuple([midpoint-range_to_keep_s[0]*30, midpoint+range_to_keep_s[1]*30])

    ctrial = cp(trial)

    for bp in ctrial.dlc_tracking['Posture'].keys():
        data = ctrial.dlc_tracking['Posture'][bp]

        if 'likelihood' in data.keys():
            data = data.drop(columns='likelihood')

        data = data.iloc[range_to_keep[0] : range_to_keep[1]]
        ctrial.dlc_tracking['Posture'][bp] = data

    return ctrial




