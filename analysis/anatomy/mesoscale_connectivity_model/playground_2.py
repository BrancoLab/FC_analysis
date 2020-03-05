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
structure_tree = cache.get_structure_tree()
SCm_id = structure_tree.get_structures_by_acronym(["SCm"])[0]["id"]
MOs_id = structure_tree.get_structures_by_acronym(["MOs"])[0]["id"]
RSP_id = structure_tree.get_structures_by_acronym(["RSP"])[0]["id"]


target = target_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=3)
source = source_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=3)
rsp_source = source_mask.get_structure_indices(structure_ids=[RSP_id], hemisphere_id=3)

# %%
sc_mask = Mask.from_cache(cache, structure_ids=[SCm_id], hemisphere_id=3)
key = sc_mask.get_key()

# %%
projection = voxel_array[source, target]
mean_proj = np.mean(projection, axis=0)


rsp_projection = voxel_array[rsp_source, target]
mrsp_ean_proj = np.mean(rsp_projection, axis=0)

mapped = sc_mask.map_masked_to_annotation(mean_proj)
rsp_mapped = sc_mask.map_masked_to_annotation(mrsp_ean_proj)


# %%
from vtkplotter import *
from vtkplotter.vtkio import loadNumpy


vol = Volume(mapped)
lego = vol.legosurface(vmin=np.mean(mapped), cmap='Greens').alpha(.5).lw(0)


vol2 = Volume(rsp_mapped)
lego2 = vol2.legosurface(vmin=np.mean(rsp_mapped), cmap='Reds').alpha(.5).lw(0)

show(lego, lego2, axes=4, viewup='z')
# %%


# %%
