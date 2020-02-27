# %%
import os
import pandas as pd
import numpy as np


from analysis.misc.paths import cellfinder_cells_folder, cellfinder_out_dir, injections_folder
from analysis.anatomy.utils import *


# %%
# Merge highest projecting regions in a summary datafame
cell_files = dict(
        # cc_136_0 = ('GRN', 'right', 'CC_136_0_ch0_cells.h5'),
        # cc_136_1 = ('GRN', 'right', 'CC_136_1_ch0_cells.h5'),
        cc_134_1 = ('SCm', 'left', 'CC_134_1_ch1_cells.h5'),
        cc_134_2 = ('SCm', 'left', 'CC_134_2_ch1_cells.h5'),
)

data = {}
df = pd.DataFrame()
ipsidf, contradf = pd.DataFrame(), pd.DataFrame()
for mouse, (inj, hemi, path) in cell_files.items():

    all_cells = pd.read_hdf(os.path.join(cellfinder_cells_folder, path), key='hdf')
    all_cells = all_cells.loc[all_cells.region != inj]

    n_cells = len(all_cells)
    threshold = 2

    ipsi = all_cells.loc[all_cells.hemisphere == hemi]
    ipsi = (ipsi.groupby('region').count().sort_values('region_name')[::-1]/ n_cells) * 100
    ipsi = ipsi.loc[ipsi.x > threshold].x.rename(f'{mouse}_{inj}_ipsi').round(2)

    contra = all_cells.loc[all_cells.hemisphere != hemi]
    contra = (contra.groupby('region').count().sort_values('region_name')[::-1]/ n_cells) * 100
    contra = contra.loc[contra.x > threshold].x.rename(f'{mouse}_{inj}_contra').round(2)
    
    
    df = pd.concat([df, ipsi, contra], axis=1).sort_index()
    ipsidf = pd.concat([ipsidf, ipsi], axis=1).sort_index()
    contradf = pd.concat([contradf, contra], axis=1).sort_index()
# print(df.to_markdown())
# %%
import networkx as nx

ipsi = ipsidf.sum(axis=1)/2
contra = contradf.sum(axis=1)/2

edges = []
regions = list(df.index)

for reg in regions:
    # try:
    #     edges.append((f'{reg}_r', 'SC_r', {'weight':ipsi[reg]}))
    # except:
    #     pass

    try:
        edges.append((f'{reg}_r', 'SC_l', {'weight':contra[reg]}))
    except:
        pass

    # try:
    #     edges.append((f'{reg}_l', 'SC_r', {'weight':contra[reg]}))
    # except:
    #     pass

    try:
        edges.append((f'{reg}_l', 'SC_l', {'weight':ipsi[reg]}))
    except:
        pass

    # edges.append((f'{reg}_l', f'{reg}_r', {'weight':1}))

G=nx.Graph()
G.add_edges_from(edges)
nx.draw(G, with_labels=True,  pos=nx.spring_layout(G))


# %%
cell_files = dict(
        cc_136_0 = ('GRN', 'right', 'CC_136_0_ch0_cells.h5'),
        cc_136_1 = ('GRN', 'right', 'CC_136_1_ch0_cells.h5'),
        # cc_134_1 = ('SCm', 'left', 'CC_134_1_ch1_cells.h5'),
        # cc_134_2 = ('SCm', 'left', 'CC_134_2_ch1_cells.h5'),
)

data = {}
df = pd.DataFrame()
ipsidf, contradf = pd.DataFrame(), pd.DataFrame()
for mouse, (inj, hemi, path) in cell_files.items():

    all_cells = pd.read_hdf(os.path.join(cellfinder_cells_folder, path), key='hdf')
    all_cells = all_cells.loc[all_cells.region != inj]

    n_cells = len(all_cells)

    ipsi = all_cells.loc[all_cells.hemisphere == hemi]
    ipsi = (ipsi.groupby('region').count().sort_values('region_name')[::-1]/ n_cells) * 100
    ipsi = ipsi.loc[ipsi.x > threshold].x.rename(f'{mouse}_{inj}_ipsi').round(2)

    contra = all_cells.loc[all_cells.hemisphere != hemi]
    contra = (contra.groupby('region').count().sort_values('region_name')[::-1]/ n_cells) * 100
    contra = contra.loc[contra.x > threshold].x.rename(f'{mouse}_{inj}_contra').round(2)
    
    
    df = pd.concat([df, ipsi, contra], axis=1).sort_index()
    ipsidf = pd.concat([ipsidf, ipsi], axis=1).sort_index()
    contradf = pd.concat([contradf, contra], axis=1).sort_index()

# %%
ipsi = ipsidf.sum(axis=1)/2
contra = contradf.sum(axis=1)/2

edges = []
regions = list(df.index)

for reg in regions:
    try:
        edges.append((f'{reg}_r', 'GRN_r', {'weight':ipsi[reg]}))
    except:
        pass

    # try:
    #     edges.append((f'{reg}_r', 'GRN_l', {'weight':contra[reg]}))
    # except:
    #     pass

    try:
        edges.append((f'{reg}_l', 'GRN_r', {'weight':contra[reg]}))
    except:
        pass

    # try:
    #     edges.append((f'{reg}_l', 'GRN_l', {'weight':ipsi[reg]}))
    # except:
    #     pass



# edges.append(('SC_r', 'SC_l', {'weight':1}))
# edges.append(('SC_l', 'GRN_r', {'weight':1}))


G.add_edges_from(edges)

# %%
nx.draw(G, with_labels=True, pos=nx.spring_layout(G))


# %%
