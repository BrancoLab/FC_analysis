# %%
import sys
import os

if sys.platform != 'darwin':
    sys.path.append('C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis')
else:
    sys.path.append('/Users/federicoclaudi/Documents/Github/FC_analysis/')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import networkx as nx

from mcmodels.core import VoxelModelCache
from mcmodels.core import Mask

from analysis.misc.paths import main_dropbox_fld
%matplotlib inline
# %%
# --------------------------------- Load data -------------------------------- #

cache_path = os.path.join(main_dropbox_fld, 'anatomy', 'mesoscale_connectome', 'voxel_model_manifest.json')
cache = VoxelModelCache(manifest_file=cache_path,
                            resolution = 100)
voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()


# %%
# Get index of array that correspond to SC targets
structure_tree = cache.get_structure_tree()

SCm_id = structure_tree.get_structures_by_acronym(["SCm"])[0]["id"]
MOs_id = structure_tree.get_structures_by_acronym(["MOs"])[0]["id"]

rSC = target_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=2)
MOs = source_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=2)


# %%
# Get the coordinates for each sc target
coords = np.array([target_mask.coordinates[p]*100 for p in rSC])

# %%
# Sort both coords and projection data
proj_mtx = voxel_array[MOs, rSC]
original = proj_mtx.copy()

sorter = np.argsort(proj_mtx)

# proj_mtx = np.sort(proj_mtx)

# f, axarr = plt.subplots(ncols=2)
# axarr[0].imshow(original)
# axarr[1].imshow(proj_mtx)


# %%
# prepr vars to create a graph
m2_nodes = [f'M2_{p}' for p in np.arange(proj_mtx.shape[0])]
sc_nodes = [f'SC_{p}' for p in np.arange(proj_mtx.shape[1])]

edges = []
th = np.mean(original) + 4 * np.std(original)
for m, m2n in enumerate(m2_nodes):
    for s, scn in enumerate(sc_nodes):
        weight = original[m, s]
        if weight > th:
            edges.append((m2n, scn, {'weight':original[m, s]}))



# %%
G=nx.DiGraph()
G.add_nodes_from(m2_nodes, region='MOs')
G.add_nodes_from(sc_nodes, region='SCm')
G.add_edges_from(edges)

isolates = nx.isolates(G)
G.remove_nodes_from(list(isolates))
G.number_of_edges()


# %%
nx.draw(G)

# %%
