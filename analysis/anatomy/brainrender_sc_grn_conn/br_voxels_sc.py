import brainrender
brainrender.SHADER_STYLE = 'cartoon'
from brainrender.Utils.parsers.mouselight import NeuronsParser
from brainrender.Utils.AllenMorphologyAPI.AllenMorphology import AllenMorphology
from brainrender.scene import Scene
from brainrender.Utils.MouseLightAPI.mouselight_info import mouselight_api_info, mouselight_fetch_neurons_metadata
from brainrender.Utils.MouseLightAPI.mouselight_api import MouseLightAPI
from brainrender.colors import colorMap
import time

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

mode = 'mean'
# sources = ['MOs', 'MOp', 'RSP', 'ZI', 'CB', 'TH', 'HY', 'ORB', 'ACA', 'PL', 'STR', 'PAG', 'CUN', 'PPN', 'SCm']
# targets = ['MOs', 'MOp', 'ZI', 'PAG', 'CUN', 'PPN', 'GRN']

# sources = ['Isocortex', ['MOs', 'MOp', 'ACA']]
sources = ['MOs', "SSs", 'SSp']
targets = ['STR']
# hemispheres= ['left', 'right']

# vapi = VolumetricAPI(add_root=False)
# for hemisphere in hemispheres:
for target in targets:
    for source in sources:
        scene_kwargs = dict(
            screenshot_kwargs = dict(
                folder='/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Locomotion/anatomy/sc_projections/screenshots',
                name=f'{source}_to_{target}_{mode}',
            ),
            use_default_key_bindings = True,
            title = f'{source} to {target}',
        )

        vapi = VolumetricAPI(add_root=False, scene_kwargs=scene_kwargs)
        vapi.render_mapped_projection(source, target,
                    std_above_mean_threshold=3,
                    # vmin=None,
                    vmax=0.0015, #0.01,
                    # hemisphere=hemisphere, 

                    cmap='gist_heat', 
                    alpha=1,
                    render_target_region=True,
                    render_source_region=False,
                    regions_kwargs={
                                'wireframe':False, 
                                'alpha':.2, 
                                'use_original_color':False},
                    projection_mode=mode,
                    mode='target',
                    )

        # vapi.scene.add_brain_regions(['VAL'], use_original_color=True, wireframe=True)

        for camera in [bespoke_camera]:
            vapi.render(interactive=False, display=False, camera=camera, zoom=1.1)  

            # time.sleep(1)
            vapi.scene.take_screenshot()

#                 # break
#         #     break
#         # break
# break
    

