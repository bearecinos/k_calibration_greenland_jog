import pandas as pd
import os
import sys
import geopandas as gpd
import numpy as np
import salem
from configobj import ConfigObj
from oggm import cfg, utils
from oggm import workflow
import warnings

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

cfg.initialize()

data_frame = []

# Reading RGI
RGI_FILE = os.path.join(MAIN_PATH, config['RGI_FILE'])
rgidf = gpd.read_file(RGI_FILE)
rgidf.crs = salem.wgs84.srs

# Select only the ice cap
# Get glaciers that belong to the ice cap.
rgidf_ice_cap = rgidf[rgidf['RGIId'].str.match('RGI60-05.10315')]
# Get the id's for filter
ice_cap_ids = rgidf_ice_cap.RGIId.values

# keeping only the Ice cap
keep_indexes = [(i in ice_cap_ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_indexes]

# Icap prepro-errors
de = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

cfg.PATHS['working_dir'] = os.path.join(MAIN_PATH, config['ice_cap_prepro_dir'])
print(cfg.PATHS['working_dir'])
cfg.PARAMS['border'] = 20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

# gdirs = workflow.init_glacier_regions(rgidf, reset=False)
gdirs = workflow.init_glacier_directories(rgidf.RGIId.values, reset=False)
print('gdirs initialized')


vbsl_no_calving_per_dir = []
ids = []
term_type = []

for gdir in gdirs:

    vbsl_no_calving_per_glacier = []

    # Get the data that we need from each glacier
    map_dx = gdir.grid.dx

    # Get flowlines
    fls = gdir.read_pickle('inversion_flowlines')

    # Get inversion output
    inv = gdir.read_pickle('inversion_output')

    import matplotlib.pylab as plt
    for f, cl, in zip(range(len(fls)), inv):
        x = np.arange(fls[f].nx) * fls[f].dx * map_dx * 1e-3
        surface = fls[f].surface_h

        # Getting the thickness per branch
        thick = cl['thick']
        vol = cl['volume']

        bed = surface - thick

        # Find volume below sea level without calving in km³
        index_sl = np.where(bed < 0.0)
        vol_sl = sum(vol[index_sl]) / 1e9
        # print('before calving',vol_sl)

        if gdir.terminus_type == 'Land-terminating':
            vol_sl = 0

        vbsl_no_calving_per_glacier = np.append(vbsl_no_calving_per_glacier,
                                                vol_sl)

    ids = np.append(ids, gdir.rgi_id)

    term_type = np.append(term_type, gdir.terminus_type)

    vbsl_no_calving_per_dir = np.append(vbsl_no_calving_per_dir,
                                        vbsl_no_calving_per_glacier.sum())

    np.set_printoptions(suppress=True)

d = {'RGIId': ids,
     'terminus_type': term_type,
     'volume bsl': vbsl_no_calving_per_dir}

data_frame = pd.DataFrame(data=d)

data_frame.to_csv(os.path.join(MAIN_PATH, config['ice_cap_prepro_dir']+
                               'volume_below_sea_level.csv'))
