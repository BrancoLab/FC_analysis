import numpy as np
import matplotlib.pyplot as plt
from brainrender.scene import Scene
from brainrender.colors import colorMap
import pandas as pd
from vtkplotter import *
from oneibl.onelight import ONE
from tqdm import tqdm


n_ch_per_probe = 374

# Fetch data
one = ONE()
one.set_figshare_url("https://figshare.com/articles/steinmetz/9974357")

# Create scene
scene = Scene()

# Get probe position for each session
sessions = one.search(['spikes'])
for n, session in tqdm(enumerate(sessions)):
    channels = one.load_object(session, 'channels')

    # Get all points
    points = dict(x=[], y=[], z=[])
    for i, p in channels.brainLocation.iterrows():
        points['x'].append(p.ccf_ap)
        points['y'].append(p.ccf_dv) 
        points['z'].append(p.ccf_lr)

    # Get points per probe
    points = pd.DataFrame(points)
    n_probes = int(len(points)/n_ch_per_probe)

    for probe in range(n_probes):
        # get points on each probe
        probe_points = points[probe * n_ch_per_probe : (probe+1) * n_ch_per_probe]
        probe_points.dropna(inplace=True)
        if not len(probe_points): continue

        # prep colors
        color = colorMap(n, name='Greens', vmin=-5, vmax=len(sessions))
        colors = [color for i in range(len(probe_points))]

        # fit line
        line = fitLine(probe_points).lw(8).alpha(0.5).color(color)

        # add actors
        # scene.add_cells(probe_points, color_by_region=False, color=colors, res=12, alpha=.5)
        scene.add_vtkactor(line)

scene.add_brain_regions(['SCm'])
scene.render()


