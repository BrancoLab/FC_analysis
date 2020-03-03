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
# --------------------------- Get projections to SC -------------------------- #
# Get mask for SC
structure_tree = cache.get_structure_tree()

SCm_id = structure_tree.get_structures_by_acronym(["SCm"])[0]["id"]
SCm_idx_r = target_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=2)
SCm_idx_l = target_mask.get_structure_indices(structure_ids=[SCm_id], hemisphere_id=1)


# Get more masks
right_hemi_idxs = target_mask.get_structure_indices(hemisphere_id=2)
left_hemi_idxs = target_mask.get_structure_indices(hemisphere_id=1)

# Get projection matrix
# a column would be the connection strength to a given voxel
# from each voxel in the right hemisphere
to_SCm_r = voxel_array[:, SCm_idx_r].T
# left_to_SC_r = voxel_array[left_hemi_idxs, SCm_idx_r].T
# right_to_SC_r = voxel_array[right_hemi_idxs, SCm_idx_r].T
to_SCm_l = voxel_array[:, SCm_idx_l].T


# %% 
# Get all structure indices for one hemisphere_id
summary_structures = structure_tree.get_structures_by_set_id([167587189])
structure_ids = [s['id'] for s in summary_structures if s['id'] != 934]

#  %%
# get projs from MOs to SC
MOs_id = structure_tree.get_structures_by_acronym(["MOs"])[0]["id"]
MOs_idx_r = source_mask.get_structure_indices(structure_ids=[MOs_id], hemisphere_id=2)


rMOs_to_SCm_r = voxel_array[MOs_idx_r, SCm_idx_r].T
lMOs_to_SCm_r = voxel_array[MOs_idx_r, SCm_idx_l].T


f, ax = plt.subplots(figsize=(20, 5), nrows=2)
ax[0].imshow(rMOs_to_SCm_r, cmap='Greens', aspect='equal', 
            interpolation='none', norm=LogNorm(vmin=0.00001, vmax=.001))
ax[1].imshow(lMOs_to_SCm_r, cmap='Greens', aspect='equal', 
            interpolation='none', norm=LogNorm(vmin=0.00001, vmax=.001))

# TODO map this stuff onto 3D visualisation
# TODO compare SC -> GRN and MOs -> GRN





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






