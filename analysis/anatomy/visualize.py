
import sys
sys.path.append('./')

import brainrender
brainrender.SHADER_STYLE='cartoon'

from brainrender.scene import Scene, DualScene 
import pandas as pd

from brainrender.colors import get_n_shades_of
from brainrender.Utils.videomaker import VideoMaker
from brainrender import *
from brainrender.Utils.data_manipulation import mirror_actor_at_point

import numpy as np
import os
from scipy.spatial.distance import euclidean
from skimage.filters import threshold_otsu
from vtkplotter.analysis import surfaceIntersection

from analysis.anatomy.utils import *
from fcutils.file_io.utils import listdir

from analysis.misc.paths import cellfinder_cells_folder, cellfinder_out_dir, injections_folder


BACKGROUND_COLOR='white'
WHOLE_SCREEN=True



class CellFinderScene(Scene):
    def add_cells_to_scene(self, cells, in_region=None, exclude_regions=None, radius=12, color_by_region=False, color='red', **kwargs):
        if in_region is not None:
            cells = get_cells_in_region(cells, in_region)

        exclude = ['ll', 'ml', 'ee', 'int', 'cc', 'ccb', 'fa', 'STR']
        if exclude_regions is not None:
            exclude.extend(list(exclude_regions))
        cells = get_cells_in_region(cells, exclude, exclude=True).dropna()
    
        if color_by_region:
            color = [list(np.float16(np.array(col)/255)) for col in self.get_region_color(list(cells.region.values))]

        self.add_cells(cells, radius=radius, color=color,  **kwargs)

    def add_injection_site(self, injection, wireframe=False, edit_kwargs=None, **kwargs):
        actor = self.add_from_file(injection, **kwargs)
        if wireframe:
            self.edit_actors([actor], wireframe=True)
        if edit_kwargs is not None:
            self.edit_actors([actor], **edit_kwargs)

        return actor

class CellFinderDoubleScene(DualScene):
    def __init__(self, *args, **kwargs):
        self.scenes = [CellFinderScene(*args, add_root=False, **kwargs),\
                        CellFinderScene(*args, add_root=True, **kwargs)]

    def add_cells_to_scenes(self, cells, in_region=[None, None], exclude_scene=None,  **kwargs):
        for i, (scene, region) in enumerate(zip(self.scenes, in_region)):
            if i != exclude_scene:
                if not region:
                    region = None
                scene.add_cells_to_scene(cells, in_region=region, **kwargs)

    def add_injection_sites(self, injection, exclude_scene=None, **kwargs):
        for i, scene in enumerate(self.scenes):
            if i != exclude_scene:
                actor = scene.add_injection_site(injection, **kwargs)
        return actor


if __name__ == "__main__":
    # ----------------------------- Visualize results CC mice ---------------------------- #q
    scene = CellFinderScene()


    grn_mice = ['CC_136_1'] #, 'CC_136_0']
    sc_mice = ['CC_134_1', 'CC_134_2']
    dario_sc_mice = ['AY_254_3', 'AY_255_1', 'AY_255_3', 'BF_172_2']

    grn_colors =[ 'salmon', 'goldenrod']
    sc_colors = get_n_shades_of('red', len(sc_mice))
    dario_sc_colors = get_n_shades_of('blue', len(dario_sc_mice))

    # ------------------------------- GRN tracings ------------------------------- #
    if False:
        for mouse, color in zip(grn_mice, grn_colors):
            cells = pd.read_hdf(os.path.join(cellfinder_cells_folder, mouse+'_ch0_cells.h5'), key='hdf')
            scene.add_cells_to_scene(cells, color=color, radius=15, res=12, alpha=.4, color_by_region=False)
                                    # in_region=['MOs', 'MOp', 'ZI', 'SCm'])

            # scene.add_injection_site(os.path.join(injections_folder, mouse+'_ch0inj.obj'), c=color)



    # -------------------------------- SC tracings ------------------------------- #
    if True:
        for mouse, color in zip(sc_mice, sc_colors):
            cells = pd.read_hdf(os.path.join(cellfinder_cells_folder, mouse+'_ch1_cells.h5'), key='hdf')

            scene.add_cells_to_scene(cells, color=color, radius=15, res=12, alpha=.4, color_by_region=False,
                                    in_region=['MOs', 'MOp', 'ACA', 'RSP', 'ZI', 'GRN'])

            scene.add_injection_site(os.path.join(injections_folder, mouse+'_ch1inj.obj'), c=color)


    # --------------------------------- dario SC --------------------------------- #
    if False:
        for mouse, color in zip(dario_sc_mice, dario_sc_colors):
            cells = pd.read_hdf(os.path.join(cellfinder_cells_folder, mouse+'_ch1_cells.h5'), key='hdf')

            scene.add_cells_to_scene(cells, color=color, radius=15, res=12, alpha=.4, color_by_region=False)
                                    # in_region=['MOs', 'MOp', 'ZI', 'GRN'])

            # scene.add_injection_site(os.path.join(injections_folder, mouse+'_ch1inj.obj'), c=color)

    # scene.add_brain_regions(['MOs', 'MOp', 'ZI'], use_original_color=True, alpha=.05,wireframe=False)
    scene.add_brain_regions(['GRN', 'SCm'], use_original_color=True, alpha=.5,wireframe=False)



    scene.render()


