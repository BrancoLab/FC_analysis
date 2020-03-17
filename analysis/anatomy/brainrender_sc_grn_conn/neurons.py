import brainrender
brainrender.SHADER_STYLE = 'cartoon'
from brainrender.Utils.parsers.mouselight import NeuronsParser
from brainrender.Utils.AllenMorphologyAPI.AllenMorphology import AllenMorphology
from brainrender.scene import Scene
from brainrender.Utils.MouseLightAPI.mouselight_info import mouselight_api_info, mouselight_fetch_neurons_metadata
from brainrender.Utils.MouseLightAPI.mouselight_api import MouseLightAPI
from brainrender.colors import colorMap

import time
import os 

from brainrender.Utils.ABA.volumetric.VolumetricConnectomeAPI import VolumetricAPI


bespoke_camera = dict(
    position = [801.843, -1339.564, 8120.729] ,
    focal = [9207.34, 2416.64, 5689.725],
    viewup = [0.36, -0.917, -0.171],
    distance = 9522.144,
    clipping = [5892.778, 14113.736],
)

zi_camera = dict(
    position = [-1482.274, 1590.553, 12053.818] ,
    focal = [7037.815, 4834.8, 5690.51],
    viewup = [0.15, -0.948, -0.282],
    distance = 11117.947,
    clipping = [5924.541, 17682.192],
)


neurons_fld = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/anatomy/sc_projections/mouselight'
neurons_files = [os.path.join(neurons_fld, f) for f in os.listdir(neurons_fld)]

grn_neurons_fld = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/anatomy/grn_projections/mouselight'
grn_neurons_files = [os.path.join(grn_neurons_fld, f) for f in os.listdir(grn_neurons_fld)]

neurons_to_both = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/anatomy/grn_projections/neurons_to_grn_and_scm.json'


target = ['SCm', 'GRN']


scene = Scene(title=f'Neurons to {target}',
                screenshot_kwargs=dict(
                    folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/anatomy/sc_projections/screenshots',
                    name = f'neurons_to_{target}',
                ))

scene.add_brain_regions(target, alpha=.4)

if target == 'SCm':
    for f in neurons_files:
        scene.add_neurons(f, color_by_region=True, 
                        render_axons=False, force_to_hemisphere='right')

if target == 'GRN':
    for f in grn_neurons_files:
        scene.add_neurons(f, color_by_region=True, 
                        render_axons=False, force_to_hemisphere='right')

if isinstance(target, list): 
    scene.add_neurons(neurons_to_both, color_by_region=True, 
                    render_axons=False, force_to_hemisphere='right')


for camera in ['top', 'three_quarters', 'sagittal']:
    scene.render(interactive = False, camera=camera, zoom=1.1)
    scene.take_screenshot()
    time.sleep(1)

