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

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
sys.path.append(MAIN_PATH)

from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Reading glacier directories per experiment
exp_dir_path = os.path.join(MAIN_PATH, config['volume_bsl_results'])
config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

full_config_paths = []
for configuration in config_paths.config_path:
    full_config_paths.append(os.path.join(MAIN_PATH, exp_dir_path,
                                          configuration + '/'))

print(full_config_paths)

# Make output dir
marcos_data = os.path.join(MAIN_PATH, 'output_data_marco')
if not os.path.exists(marcos_data):
    os.makedirs(marcos_data)

cfg.initialize()

data_frame = []

for path, output_config in zip(full_config_paths, config_paths.results_output):

    experimet_name = misc.splitall(path)[-2]
    exp_dir_output = os.path.join(marcos_data, experimet_name)

    if not os.path.exists(exp_dir_output):
        os.makedirs(exp_dir_output)

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

    no_solution = os.path.join(output_path, 'glaciers_with_no_solution.csv')
    d_no_sol = pd.read_csv(no_solution)
    ids_rgi = d_no_sol.RGIId.values
    keep_no_solution = [(i not in ids_rgi) for i in rgidf.RGIId]
    rgidf = rgidf.iloc[keep_no_solution]

    # Remove glaciers with no input data
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

    # gdirs = workflow.init_glacier_regions(rgidf, reset=False)
    gdirs = workflow.init_glacier_directories(rgidf.RGIId.values, reset=False)
    print('gdirs initialized')

    for gdir in gdirs:

        # Get inversion output
        inv_c = gdir.read_pickle('inversion_output')[-1]
        surface = gdir.read_pickle('inversion_flowlines')[-1].surface_h
        diags = gdir.get_diagnostics()
        water_depth = diags['calving_front_water_depth']
        free_board = diags['calving_front_free_board']
        calving_flux = diags['calving_flux']
        depths = np.zeros(len(inv_c['thick'][-5:]))
        board = np.zeros(len(inv_c['thick'][-5:]))
        flux = np.zeros(len(inv_c['thick'][-5:]))
        depths[-1:] = water_depth
        board[-1:] = free_board
        flux[-1:] = calving_flux

        d = {'thick_end_fls': inv_c['thick'][-5:],
             'width_end_fls': inv_c['width'][-5:],
             'is_rectangular': inv_c['is_rectangular'][-5:],
             'slope': inv_c['slope_angle'][-5:],
             'elevation [m]': surface[-5:],
             'calving_front_water_depth': depths,
             'calving_front_free_board': board,
             'calving_flux': flux}

        data_frame = pd.DataFrame(data=d)

        data_frame.to_csv(os.path.join(exp_dir_output, gdir.rgi_id + '.csv'))
