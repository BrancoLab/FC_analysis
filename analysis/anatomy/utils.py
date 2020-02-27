import pandas as pd
import os

from brainrender.Utils.ABA.connectome import ABA

from fcutils.file_io.io import save_yaml, load_yaml
from fcutils.file_io.utils import listdir

from analysis.misc.paths import brainsaw_folder, cellfinder_cells_folder, cellfinder_out_dir, injections_folder

aba = ABA()

regions_summary_filepath = os.path.join(cellfinder_out_dir, 'regions_summary.hdf')

# ---------------------------------------------------------------------------- #
#                               GET FILES FUNCTIO                              #
# ---------------------------------------------------------------------------- #
def get_injection_site_for_mouse(mouse, ch=1):
    correct = [f for f in listdir(injections_folder) if mouse in f and 'ch{}'.format(ch) in f]
    if not correct:
        raise FileNotFoundError("COuld not find injection file for mouse "+mouse)
    elif len(correct) > 1:
        raise ValueError("Found too many files!")
    else:
        return correct[0]

def get_cells_for_mouse(mouse, ch=1):
    correct = [f for f in listdir(cellfinder_cells_folder) if mouse in f and 'ch{}'.format(ch) in f]
    if not correct:
        print("Could not find cells file for mouse "+mouse)
        return None
    elif len(correct) > 1:
        raise ValueError("Found too many files!!")
    else:
        return pd.read_hdf(correct[0], key='hdf')

def get_mice():
    return sorted(set([f.split("_ch")[0] for f in os.listdir(cellfinder_cells_folder)]))


# ---------------------------------------------------------------------------- #
#                            OTHER UTILITY FUNCTIONS                           #
# ---------------------------------------------------------------------------- #
def get_count_by_brain_region(cells):
    n_cells = len(cells)
    count_per_region =  cells.groupby('region').count().sort_values('type', ascending=False)['type']
    return n_cells, count_per_region, count_per_region/n_cells

def get_cells_in_region(cells, region, exclude=False):
    if isinstance(region, list):
        region_list = []
        for reg in region:
            region_list.extend(list(aba.get_structure_descendants(reg)['acronym'].values))
    else:
        region_list =  list(aba.get_structure_descendants(region)['acronym'].values)
    
    if not exclude:
        return cells[cells.region.isin(region_list)]
    else:
        return cells[~cells.region.isin(region_list)]
        


# ---------------------------------------------------------------------------- #
#                           ACRONYMS OF BRAIN REGIONS                          #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    scm = list(aba.get_structure_descendants('SCm')['acronym'].values)
    scs = list(aba.get_structure_descendants('SCs')['acronym'].values)
    mos = list(aba.get_structure_descendants('MOs')['acronym'].values)
    mop = list(aba.get_structure_descendants('MOp')['acronym'].values)
    rsc = list(aba.get_structure_descendants('RSP')['acronym'].values)
    zi = list(aba.get_structure_descendants('ZI')['acronym'].values)
    ssp = list(aba.get_structure_descendants('SSp')['acronym'].values)
    sss = list(aba.get_structure_descendants('SSs')['acronym'].values)
    visp = list(aba.get_structure_descendants('VISp')['acronym'].values)
    ic = list(aba.get_structure_descendants('IC')['acronym'].values)
    cerebellum = list(aba.get_structure_descendants('CB')['acronym'].values)
    aud = list(aba.get_structure_descendants('AUD')['acronym'].values)
    ptlp = list(aba.get_structure_descendants('PTLp')['acronym'].values)
    aca = list(aba.get_structure_descendants('ACA')['acronym'].values)
    ent = list(aba.get_structure_descendants('ENT')['acronym'].values)
    tea = list(aba.get_structure_descendants('TEa')['acronym'].values)

    all_regions = dict(MOs=mos, MOp=mop, RSP=rsc, ZI=zi,
                    SSp=ssp, SSs=sss, VISp=visp, SCm=scm, 
                    SCs=scs, IC=ic, CB=cerebellum, ENT=ent,
                    AUD=aud, PTLp=ptlp, ACA=aca, TEa=tea)

    save_yaml('Analysis/anatomy/acronyms.yaml', all_regions)

    # Get a condensed list of all regions acronyms
    all_regions_acronyms = list(aba.get_structure_descendants('root')['acronym'].values)

    clean_all_regions_acronyms = []
    for acronym in all_regions_acronyms:
        acro = False
        for region_name, acronyms in all_regions.items():
            if acronym in acronyms: 
                clean_all_regions_acronyms.append(region_name)
                acro = True
                break
        if not acro:
            clean_all_regions_acronyms.append(acronym)

    all_regions_acronyms = clean_all_regions_acronyms.copy()
    save_yaml('Analysis/anatomy/all_acronyms.yaml', all_regions_acronyms)
else:
    all_regions = load_yaml('analysis/anatomy/acronyms.yaml')
    all_regions_acronyms = load_yaml('analysis/anatomy/all_acronyms.yaml')


regions_to_exclude =  ['ll', 'ml',  'ee', 'int', 'cc', 'ccb', 'fa', 'f']

__all__ = [
    "aba",
    "cellfinder_cells_folder",
    "injections_folder",
    "cellfinder_out_dir",
    "get_injection_site_for_mouse", 
    "get_cells_for_mouse",
    "get_mice",
    "get_count_by_brain_region",
    "all_regions",
    "get_cells_in_region",
    "regions_summary_filepath",
    "all_regions_acronyms",
    "regions_to_exclude",
]