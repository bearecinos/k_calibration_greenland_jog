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

# Reading glacier directories per experiment
exp_dir_path = os.path.join(MAIN_PATH, config['volume_bsl_results'])
config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

print(config_paths.config_path)
print(config_paths.results_output)

full_config_paths = []
for configuration in config_paths.config_path:
    full_config_paths.append(os.path.join(MAIN_PATH, exp_dir_path,
                                          configuration + '/'))

cfg.initialize()

data_frame = []

for path, output_config in zip(full_config_paths, config_paths.results_output):

    experimet_name = misc.splitall(path)[-2]

    # Reading RGI
    RGI_FILE = os.path.join(MAIN_PATH, config['RGI_FILE'])
    rgidf = gpd.read_file(RGI_FILE)
    rgidf.crs = salem.wgs84.srs


    # Select only Marine-terminating
    glac_type = ['0']
    keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
    rgidf = rgidf.iloc[keep_glactype]

    connection = [2]
    keep_connection = [(i not in connection) for i in rgidf.Connect]
    rgidf = rgidf.iloc[keep_connection]

    # Remove pre-pro errors
    de = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']))
    ids = de.RGIId.values
    keep_errors = [(i not in ids) for i in rgidf.RGIId]
    rgidf = rgidf.iloc[keep_errors]

    # Remove glaciers with no solution

    output_path = os.path.join(MAIN_PATH, config[output_config])
    print(output_path)

    no_solution = os.path.join(output_path, 'glaciers_with_no_solution.csv')
    d_no_sol = pd.read_csv(no_solution)
    ids_rgi = d_no_sol.RGIId.values
    keep_no_solution = [(i not in ids_rgi) for i in rgidf.RGIId]
    rgidf = rgidf.iloc[keep_no_solution]

    no_data = os.path.join(output_path,
                           'glaciers_with_no_vel_data.csv')
    if os.path.exists(no_data):
        no_data = no_data
    else:
        no_data = os.path.join(output_path,
                               'glaciers_with_no_racmo_data.csv')

    d_no_data = pd.read_csv(no_data)
    ids_no_data = d_no_data.RGIId.values
    keep_no_data = [(i not in ids_no_data) for i in rgidf.RGIId]
    rgidf = rgidf.iloc[keep_no_data]

    cfg.PATHS['working_dir'] = path
    print(cfg.PATHS['working_dir'])
    cfg.PARAMS['border'] = 20
    cfg.PARAMS['use_tar_shapefiles'] = False
    cfg.PARAMS['use_intersects'] = True
    cfg.PARAMS['use_compression'] = False
    cfg.PARAMS['compress_climate_netcdf'] = False

    #
    # gdirs = workflow.init_glacier_regions(rgidf, reset=False)
    gdirs = workflow.init_glacier_directories(rgidf.RGIId.values, reset=False)
    print('gdirs initialized')

    vbsl_no_calving_per_dir = []
    vbsl_calving_per_dir = []
    ids = []

    for gdir in gdirs:

        vbsl_no_calving_per_glacier = []
        vbsl_calving_per_glacier = []

        # Get the data that we need from each glacier
        map_dx = gdir.grid.dx

        # Get flowlines
        fls = gdir.read_pickle('inversion_flowlines')

        # Get inversion output
        inv = gdir.read_pickle('inversion_output',
                               filesuffix='_without_calving_')
        inv_c = gdir.read_pickle('inversion_output')

        import matplotlib.pylab as plt

        for f, cl, cc, in zip(range(len(fls)), inv, inv_c):
            x = np.arange(fls[f].nx) * fls[f].dx * map_dx * 1e-3
            surface = fls[f].surface_h

            # Getting the thickness per branch
            thick = cl['thick']
            vol = cl['volume']

            thick_c = cc['thick']
            vol_c = cc['volume']

            bed = surface - thick
            bed_c = surface - thick_c

            # Find volume below sea level without calving in km³
            index_sl = np.where(bed < 0.0)
            vol_sl = sum(vol[index_sl]) / 1e9
            # print('before calving',vol_sl)

            index_sl_c = np.where(bed_c < 0.0)
            vol_sl_c = sum(vol_c[index_sl_c]) / 1e9
            # print('after calving',vol_sl_c)

            vbsl_no_calving_per_glacier = np.append(
                vbsl_no_calving_per_glacier, vol_sl)

            vbsl_calving_per_glacier = np.append(
                vbsl_calving_per_glacier, vol_sl_c)

            ids = np.append(ids, gdir.rgi_id)

        # We sum up all the volume below sea level in all branches
        vbsl_no_calving_per_glacier = sum(vbsl_no_calving_per_glacier)
        vbsl_calving_per_glacier = sum(vbsl_calving_per_glacier)

        vbsl_no_calving_per_dir = np.append(vbsl_no_calving_per_dir,
                                            vbsl_no_calving_per_glacier)

        vbsl_calving_per_dir = np.append(vbsl_calving_per_dir,
                                         vbsl_calving_per_glacier)

        np.set_printoptions(suppress=True)

    d = {'RGIId': pd.unique(ids),
         'volume bsl': vbsl_no_calving_per_dir,
         'volume bsl with calving': vbsl_calving_per_dir}

    data_frame = pd.DataFrame(data=d)

    data_frame.to_csv(os.path.join(path, experimet_name +
                                   '_volume_below_sea_level.csv'))
