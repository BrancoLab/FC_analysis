# %%
# https://mouse-connectivity-models.readthedocs.io/en/latest/modules/voxel.html
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
# -------------------------------- Get indexes ------------------------------- #
# Get atlas ID numbers
structure_tree = cache.get_structure_tree()
SCm_id = structure_tree.get_structures_by_acronym(["SCm"])[0]["id"]
GRN_id = structure_tree.get_structures_by_acronym(["GRN"])[0]["id"]
MOs_id = structure_tree.get_structures_by_acronym(["MOs"])[0]["id"]
MOp_id = structure_tree.get_structures_by_acronym(["MOp"])[0]["id"]
RSP_id = structure_tree.get_structures_by_acronym(["RSP"])[0]["id"]
ZI_id = structure_tree.get_structures_by_acronym(["ZI"])[0]["id"]

MOs_layers_id = [
    structure_tree.get_structures_by_acronym(["MOs5"])[0]["id"],
    structure_tree.get_structures_by_acronym(["MOs6a"])[0]["id"],
    structure_tree.get_structures_by_acronym(["MOs6b"])[0]["id"],
]

MOp_layers_id = [
    structure_tree.get_structures_by_acronym(["MOp5"])[0]["id"],
    structure_tree.get_structures_by_acronym(["MOp6a"])[0]["id"],
    structure_tree.get_structures_by_acronym(["MOp6b"])[0]["id"],
]

RSP_layers_id = [
    structure_tree.get_structures_by_acronym(["RSPagl5"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPd5"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPv5"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPagl6a"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPd6a"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPv6a"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPagl6b"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPd6b"])[0]["id"],
    structure_tree.get_structures_by_acronym(["RSPv6b"])[0]["id"],
]

summary_structures = structure_tree.get_structures_by_set_id([167587189])
structure_ids = [s['id'] for s in summary_structures if s['id'] != 934]

# Get voxel indices
sources = {}
targets = {'right':{}, 'left':{}}

targets['right']['SCm'] = target_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=2)
targets['left']['SCm'] = target_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=1)
targets['right']['GRN'] = target_mask.get_structure_indices(structure_ids=[GRN_id], hemisphere_id=2)
targets['left']['GRN'] = target_mask.get_structure_indices(structure_ids=[GRN_id], hemisphere_id=1)
targets['left']['MOs'] = target_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=1)
targets['right']['MOs'] = target_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=2)
targets['left']['MOs_layers'] = target_mask.get_structure_indices(structure_ids=MOs_layers_id, hemisphere_id=1)
targets['right']['MOs_layers'] = target_mask.get_structure_indices(structure_ids=MOs_layers_id, hemisphere_id=2)




sources['MOp'] = source_mask.get_structure_indices(structure_ids=[MOp_id], hemisphere_id=2)
sources['MOp_layers'] = source_mask.get_structure_indices(structure_ids=MOp_layers_id, hemisphere_id=2)
sources['MOs'] = source_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=2)
sources['MOs_layers'] = source_mask.get_structure_indices(structure_ids=MOs_layers_id, hemisphere_id=2)
sources['RSP'] = source_mask.get_structure_indices(structure_ids=[RSP_id], hemisphere_id=2)
sources['RSP_layers'] = source_mask.get_structure_indices(structure_ids=RSP_layers_id, hemisphere_id=2)
sources['ZI'] = source_mask.get_structure_indices(structure_ids=[ZI_id], hemisphere_id=2)
sources['SCm'] = source_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=2)


# %%
# ------------------------------- Get proj mtxs ------------------------------ #
mop_to_lSC = voxel_array[sources['MOp_layers'], targets['left']['SCm']].T
mop_to_rSC = voxel_array[sources['MOp_layers'], targets['right']['SCm']].T
mop_to_lGRN = voxel_array[sources['MOp_layers'], targets['left']['GRN']].T
mop_to_rGRN = voxel_array[sources['MOp_layers'], targets['right']['GRN']].T

mos_to_lSC = voxel_array[sources['MOs_layers'], targets['left']['SCm']].T
mos_to_rSC = voxel_array[sources['MOs_layers'], targets['right']['SCm']].T
mos_to_lGRN = voxel_array[sources['MOs_layers'], targets['left']['GRN']].T
mos_to_rGRN = voxel_array[sources['MOs_layers'], targets['right']['GRN']].T

rsp_to_lSC = voxel_array[sources['RSP_layers'], targets['left']['SCm']].T
rsp_to_rSC = voxel_array[sources['RSP_layers'], targets['right']['SCm']].T
rsp_to_lGRN = voxel_array[sources['RSP_layers'], targets['left']['GRN']].T
rsp_to_rGRN = voxel_array[sources['RSP_layers'], targets['right']['GRN']].T

zi_to_lSC = voxel_array[sources['ZI'], targets['left']['SCm']].T
zi_to_rSC = voxel_array[sources['ZI'], targets['right']['SCm']].T
zi_to_lGRN = voxel_array[sources['ZI'], targets['left']['GRN']].T
zi_to_rGRN = voxel_array[sources['ZI'], targets['right']['GRN']].T

# %% 
# ----------------------------- Show projections ----------------------------- #

# Get sorting
m2_sort_idx = np.argsort(mos_to_rSC.mean(axis=0))
lsc_sort = np.argsort(mos_to_lSC.mean(axis=1))
rsc_sort = np.argsort(mos_to_rSC.mean(axis=1))

f, axarr = plt.subplots(ncols=8, nrows=2, figsize=(18, 6), sharex=False)

# MOp -> SC
axarr[0, 0].imshow(mop_to_lSC, 
            cmap='Greens', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[0, 1].imshow(mop_to_rSC, 
            cmap='Greens', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

# MOp -> GRN
axarr[1, 0].imshow(mop_to_lGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[1, 1].imshow(mop_to_rGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

# MOs -> SC
axarr[0, 2].imshow(mos_to_lSC, 
            cmap='Greens', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[0, 3].imshow(mos_to_rSC[rsc_sort, :], 
            cmap='Greens', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

# MOs -> GRN
axarr[1, 2].imshow(mos_to_lGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[1, 3].imshow(mos_to_rGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))


# RSP -> SC
axarr[0, 4].imshow(rsp_to_lSC, 
            cmap='Purples', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[0, 5].imshow(rsp_to_rSC, 
            cmap='Purples', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

# RSP -> GRN
axarr[1, 4].imshow(rsp_to_lGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[1, 5].imshow(rsp_to_rGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

# ZI -> SC
axarr[0, 6].imshow(zi_to_lSC, 
            cmap='Reds', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[0, 7].imshow(zi_to_rSC, 
            cmap='Reds', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

# ZI -> GRN
axarr[1, 6].imshow(zi_to_lGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))

axarr[1, 7].imshow(zi_to_rGRN, 
            cmap='Greys', interpolation='none', 
            norm=LogNorm(vmin=0.00001, vmax=.004))



axarr[0,0].set(title='MOp to left SCm', ylabel='SCm voxels')
axarr[0,1].set(title='MOp to right SCm')
axarr[1,0].set(title='MOp to left GRN', ylabel='GRN voxels', xlabel='MOp voxels')
axarr[1,1].set(title='MOp to right GRN', xlabel='MOp voxels')

axarr[0,2].set(title='MOs to left SCm', ylabel='SCm voxels')
axarr[0,3].set(title='MOs to right SCm')
axarr[1,2].set(title='MOs to left GRN', ylabel='GRN voxels', xlabel='MOs voxels')
axarr[1,3].set(title='MOs to right GRN', xlabel='MOs voxels')

axarr[0,4].set(title='RSP to left SCm')
axarr[0,5].set(title='RSP to right SCm')
axarr[1,4].set(title='RSP to left GRN', xlabel='RSP voxels')
axarr[1,5].set(title='RSP to right GRN', xlabel='RSP voxels')

axarr[0,6].set(title='ZI to left SCm')
axarr[0,7].set(title='ZI to right SCm')
axarr[1,6].set(title='ZI to left GRN', xlabel='ZI voxels')
axarr[1,7].set(title='ZI to right GRN', xlabel='ZI voxels')

f.tight_layout()
# TODO map this stuff onto 3D visualisation


# %%
# Construct adjacency mtx for MOs and SC
mts = voxel_array[sources['MOs_layers'], targets['right']['SCm']].T
sts = voxel_array[sources['SCm'], targets['right']['SCm']].T
mtm = voxel_array[sources['MOs_layers'], targets['right']['MOs_layers']].T
stm = voxel_array[sources['SCm'], targets['right']['MOs_layers']].T

top = np.hstack([mts, sts])
bottom = np.hstack([mtm, stm])
adj = np.vstack([top, bottom])

# %%
# ------------------- Construct graph from adjacenty matrix ------------------ #

G=nx.from_numpy_matrix(adj)
nx.draw(G)


# %%

mask = Mask.from_cache(cache, structure_ids=[SCm_id])

mask.get_flattened_voxel_index(0)

# import napari


# with napari.gui_qt():
#     # create the viewer with an image
#     viewer = napari.view_image(mask.mask, rgb=False)