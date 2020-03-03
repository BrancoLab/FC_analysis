# %%
# https://mouse-connectivity-models.readthedocs.io/en/latest/modules/voxel.html
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mcmodels.core import VoxelModelCache
from mcmodels.core import Mask

# %%
# --------------------------------- Load data -------------------------------- #
cache = VoxelModelCache(manifest_file='/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Anatomy/mesoscale_connectome/voxel_model_manifest.json',
                            resolution = 100)
voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()


# %%
# -------------------------------- Get indexes ------------------------------- #
# Get atlas ID numbers
structure_tree = cache.get_structure_tree()
SCm_id = structure_tree.get_structures_by_acronym(["SCm"])[0]["id"]
GRN_id = structure_tree.get_structures_by_acronym(["GRN"])[0]["id"]
MOs_id = structure_tree.get_structures_by_acronym(["MOs"])[0]["id"]
RSP_id = structure_tree.get_structures_by_acronym(["RSP"])[0]["id"]

MOs_layers_id = [
    structure_tree.get_structures_by_acronym(["MOs5"])[0]["id"],
    structure_tree.get_structures_by_acronym(["MOs6a"])[0]["id"],
    structure_tree.get_structures_by_acronym(["MOs6b"])[0]["id"],
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

sources['MOs'] = source_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=2)
sources['MOs_layers'] = source_mask.get_structure_indices(structure_ids=MOs_layers_id, hemisphere_id=2)
sources['RSP'] = source_mask.get_structure_indices(structure_ids=[RSP_id], hemisphere_id=2)
sources['RSP_layers'] = source_mask.get_structure_indices(structure_ids=RSP_layers_id, hemisphere_id=2)

# %%
# ------------------------------- Get proj mtxs ------------------------------ #
mos_to_lSC = voxel_array[sources['MOs_layers'], targets['left']['SCm']].T
mos_to_rSC = voxel_array[sources['MOs_layers'], targets['right']['SCm']].T

rsp_to_lSC = voxel_array[sources['RSP_layers'], targets['left']['SCm']].T
rsp_to_rSC = voxel_array[sources['RSP_layers'], targets['right']['SCm']].T

mos_to_lGRN = voxel_array[sources['MOs_layers'], targets['left']['GRN']].T
mos_to_rGRN = voxel_array[sources['MOs_layers'], targets['right']['GRN']].T

# Get sorting
sc_sort_idx = np.argsort(mos_to_rSC, axis=0)

# %% 
# ----------------------------- Show projections ----------------------------- #
f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(10, 6), sharex=True)

# MOs -> SC
axarr[0, 0].imshow(mos_to_rSC[sc_sort_idx], 
            cmap='Greens', interpolation='none', )
            # norm=LogNorm(vmin=0.00001, vmax=.001))

axarr[0, 1].imshow(mos_to_rSC[sc_sort_idx], 
            cmap='Greens', interpolation='none', )
            # norm=LogNorm(vmin=0.00001, vmax=.001))

# MOs -> GRN
axarr[1, 0].imshow(mos_to_lGRN, 
            cmap='Reds', interpolation='none', )
            # norm=LogNorm(vmin=0.00001, vmax=.001))

axarr[1, 1].imshow(mos_to_rGRN, 
            cmap='Reds', interpolation='none', )
            # norm=LogNorm(vmin=0.00001, vmax=.001))

# RSP -> SC
axarr[0, 2].imshow(rsp_to_lSC[sc_sort_idx], 
            cmap='Purples', interpolation='none', )
            # norm=LogNorm(vmin=0.00001, vmax=.001))

axarr[0, 3].imshow(rsp_to_rSC[sc_sort_idx], 
            cmap='Purples', interpolation='none', )
            # norm=LogNorm(vmin=0.00001, vmax=.001))


axarr[0,0].set(title='Right deep MOs to left SCm', ylabel='SCm voxels')
axarr[0,1].set(title='Right deep MOs to right SCm')
axarr[1,0].set(title='Right deep MOs to left GRN', ylabel='GRN voxels', xlabel='deep MOs voxels')
axarr[1,1].set(title='Right deep MOs to right GRN', xlabel='deep MOs voxels')
axarr[0,2].set(title='Right deep RSP to left SCm', ylabel='SCm voxels')
axarr[0,3].set(title='Right deep RSP to right SCm')

# TODO map this stuff onto 3D visualisation


# %%
# ----------------------- Map back to volume and render ---------------------- #
mask = Mask.from_cache(cache)




# %%
"""
https://mouse-connectivity-models.readthedocs.io/en/latest/modules/generated/mcmodels.core.VoxelModelCache.html
    Potentially useful functions
    cache.get_affine_parameters
    cache.get_annotation_volume
    cache.get_connection_strength
    cache.get_normalized_connection_strength

    cache.get_structure_mask
    cache.get_structure_mesh
"""






