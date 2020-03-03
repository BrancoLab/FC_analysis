# %%

import sys
import os
import pandas as pd
if sys.platform != 'darwin':
    sys.path.append('C:\\Users\\Federico\\Documents\\GitHub\\FC_analysis')
else:
    sys.path.append('/Users/federicoclaudi/Documents/Github/FC_analysis/')
from analysis.misc.paths import main_dropbox_fld

from mcmodels.core import VoxelModelCache, VoxelData
from mcmodels.models.voxel import RegionalizedModel

from fcutils.plotting.colors import *
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.utils import save_figure

# %%

cache_path = os.path.join(main_dropbox_fld, 'anatomy', 'mesoscale_connectome', 'voxel_model_manifest.json')
cache = VoxelModelCache(manifest_file=cache_path)

normalized_connection_density = cache.get_normalized_connection_density()

structure_tree = cache.get_structure_tree()
summary_structures = pd.DataFrame(structure_tree.get_structures_by_set_id([167587189]))

ipsi = normalized_connection_density['ipsi']
contra = normalized_connection_density['contra']

for df in [ipsi, contra]:
    # Rename cols
    structures_acros = [summary_structures.loc[summary_structures['id'] == int(col)].acronym.values[0]
                                for col in df.columns]
    new_cols = {c:nc for c, nc in zip(df.columns, structures_acros)}
    df.rename(columns=new_cols, inplace=True)

    # rename rows
    structures_acros = [summary_structures.loc[summary_structures['id'] == int(col)].acronym.values[0]
                                for col in df.index]
    new_idx = {c:nc for c, nc in zip(df.index, structures_acros)}
    df.rename(columns=new_cols, index=new_idx, inplace=True)
ipsi

# %%
# Whole amtrix
f, axarr = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(48, 24))

axarr[0].imshow(ipsi.values, vmax=0.001, cmap='Greens')
axarr[1].imshow(contra.values, vmax=0.001, cmap='Greens')

_ = axarr[0].set(title='IPSI', ylabel='source', yticks=np.arange(len(ipsi)),
                    yticklabels=ipsi.index, xticks=np.arange(ipsi.values.shape[1]),
                    xticklabels=ipsi.columns)

save_figure(f, 'mtx')
# %%
# Projections to target
import seaborn as sns

f, axarr = plt.subplots(ncols=1, nrows=2, figsize=(40, 20), sharex=True)

axarr[0].bar(ipsi.index, ipsi['SCm'], color=skyblue)
axarr[0].bar(contra.index, -contra['SCm'], color=desaturate_color(skyblue, k=.3))

axarr[1].bar(ipsi.index, ipsi['GRN'], color=salmon)
axarr[1].bar(contra.index, -contra['GRN'], color=desaturate_color(salmon, k=.3))

axarr[0].set(title='Proj to SCm')
axarr[1].set(title='Proj to GRN')

plt.xticks(rotation=90, fontsize=4)

save_figure(f, 'projs', svg=True)
# %%
