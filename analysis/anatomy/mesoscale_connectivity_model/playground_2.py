# %%
from mcmodels.core import VoxelModelCache
from mcmodels.models.voxel import RegionalizedModel
cache = VoxelModelCache()


# pull voxel-scale model from cache
voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
# regionalize to summary structures (region 934 was removed in new ccf)
regions = cache.get_structures_by_set_id()
region_ids = [r['id'] for r in regions if r['id'] != 934]

# get array keys
source_key = source_mask.get_key(region_ids)
target_key = source_mask.get_key(region_ids)
# regionalize model
regional_model = RegionalizedModel.from_voxel_array(voxel_array, source_key, target_key)
regional_model.normalized_connection_density.shape
(291, 577)