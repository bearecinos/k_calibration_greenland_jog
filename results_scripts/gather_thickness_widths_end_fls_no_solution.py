import pandas as pd
import os
import sys
import geopandas as gpd
import numpy as np
import salem
from configobj import ConfigObj
from oggm import cfg, utils
from oggm import workflow
import glob
import argparse

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']
input_data_path = config['input_data_folder']
sys.path.append(MAIN_PATH)

from k_tools import misc

# Reading glacier directory from prepro
exp_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Glaciers_no_solution')

# Reading thickness data form Millan et all 2022
path_h = os.path.join(MAIN_PATH, config['thickness_obs'])

#Read freeboard and width from glacier stats
path_stats = os.path.join(exp_dir_path, 'glacier_statistics_greenland_calving_with_sliding.csv')
df_stats = pd.read_csv(path_stats)

# Make output dir
marcos_data = os.path.join(MAIN_PATH, 'output_data_marco')
if not os.path.exists(marcos_data):
    os.makedirs(marcos_data)

cfg.initialize()

data_frame = []

experimet_name = misc.splitall(exp_dir_path)[-1]
exp_dir_output = os.path.join(marcos_data, experimet_name)

print(experimet_name)
print(exp_dir_output)


if not os.path.exists(exp_dir_output):
    os.makedirs(exp_dir_output)

# Reading RGI
RGI_FILE = os.path.join(input_data_path, config['RGI_FILE'])
rgidf = gpd.read_file(RGI_FILE)
rgidf.crs = salem.wgs84.srs


# Get glaciers that belong to the ice cap.
rgidf_ice_cap = rgidf[rgidf['RGIId'].str.match('RGI60-05.10315')]
# Get the id's for filter
ice_cap_ids = rgidf_ice_cap.RGIId.values

# Removing the Ice cap
keep_indexes = [(i not in ice_cap_ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_indexes]

# Select only Marine-terminating
glac_type = ['0']
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Remove pre-pro errors
de = pd.read_csv(os.path.join(input_data_path, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

# Remove glaciers with no solution
output_path = os.path.join(MAIN_PATH, config['vel_calibration_results_measures'])

no_solution = os.path.join(output_path, 'glaciers_with_no_solution.csv')
d_no_sol = pd.read_csv(no_solution)
ids_rgi = d_no_sol.RGIId.values
keep_no_solution = [(i in ids_rgi) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_solution]

cfg.PATHS['working_dir'] = exp_dir_path
print(cfg.PATHS['working_dir'])
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['mp_processes'] = 16
cfg.PARAMS['border'] = 20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False
cfg.PARAMS['use_kcalving_for_inversion'] = True
cfg.PARAMS['use_kcalving_for_ru'] = True

# gdirs = workflow.init_glacier_regions(rgidf, reset=False)
gdirs = workflow.init_glacier_directories(rgidf.RGIId.values, reset=False)
print('gdirs initialized')

for gdir in gdirs:
   
    index_stats = df_stats.index[df_stats['rgi_id'] == gdir.rgi_id].tolist()
    data_oggm = df_stats.iloc[index_stats]
    
    # Get inversion output
    inv_c = gdir.read_pickle('inversion_output')[-1]
    surface = gdir.read_pickle('inversion_flowlines')[-1].surface_h
    free_board = data_oggm['calving_front_free_board'].values
    board = np.zeros(len(inv_c['thick']))
    board[-1:] = free_board

    d = {'thick_end_fls': inv_c['thick'],
         'width_end_fls': inv_c['width'],
         'is_rectangular': inv_c['is_rectangular'],
         'slope': inv_c['slope_angle'],
         'elevation [m]': surface,
         'calving_front_free_board': board}

    data_frame = pd.DataFrame(data=d)

    h_obs_path = os.path.join(path_h, gdir.rgi_id+'.csv')
    if os.path.exists(h_obs_path):
        glacier_obs = pd.read_csv(h_obs_path)
        data_frame = pd.concat([data_frame, glacier_obs], axis=1)
    print(exp_dir_output)
    data_frame.to_csv(os.path.join(exp_dir_output, gdir.rgi_id + '.csv'))
