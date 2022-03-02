# This will run OGGM and obtain thickness data from Millan et al. 2022
# It will give you thickness data at the last pixel of the flowline
# and along the main centerline, data its store as a .csv file
# for each glacier in the main working directory

# Python imports
from __future__ import division
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from configobj import ConfigObj
import time
import salem
import argparse

# Imports oggm
import oggm.cfg as cfg
from oggm import workflow, utils
from oggm import tasks
from oggm.workflow import execute_entity_task

# Module logger
import logging
log = logging.getLogger(__name__)
# Time
start = time.time()

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-mode", type=bool, default=False, help="pass running mode")
args = parser.parse_args()
config_file = args.conf
run_mode = args.mode

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']
input_data_path = config['input_data_folder']
sys.path.append(MAIN_PATH)

# velocity module
from k_tools import utils_thick as utils_h
from k_tools import misc

# Regions: Greenland
rgi_region = '05'
rgi_version = '61'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------
cfg.initialize()

# Define working directories (either local if run_mode = true)
# or in the cluster environment
if run_mode:
    cfg.PATHS['working_dir'] = utils.get_temp_dir('GP-test-run')
else:
    SLURM_WORKDIR = os.environ["OUTDIR"]
    # Local paths (where to write output and where to download input)
    WORKING_DIR = SLURM_WORKDIR
    cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing
if run_mode:
    cfg.PARAMS['use_multiprocessing'] = False
else:
    # ONLY IN THE CLUSTER!
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 16

cfg.PARAMS['border'] = 20
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['min_mu_star'] = 0.0
cfg.PARAMS['clip_mu_star'] = True
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False
cfg.PARAMS['clip_tidewater_border'] = False

# RGI file
rgidf = gpd.read_file(os.path.join(input_data_path, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# We use intersects
cfg.set_intersects_db(os.path.join(input_data_path, config['intercepts']))
rgidf = rgidf.sort_values('RGIId', ascending=True)

if not run_mode:
    # Read Areas for the ice-cap computed in OGGM during
    # the pre-processing runs
    df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                            config['ice_cap_prepro']))

    df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

    # Assign an area to the ice cap from OGGM to avoid errors
    rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
              'Area'] = df_prepro_ic.rgi_area_km2.values

# Run only for Lake Terminating and Marine Terminating
glac_type = ['0']
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Run only glaciers that have a week connection or are
# not connected to the ice-sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Exclude glaciers with prepro-erros
de = pd.read_csv(os.path.join(input_data_path, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

# Remove glaciers that need to be model with gimp
df_gimp = pd.read_csv(os.path.join(input_data_path, config['glaciers_gimp']))
keep_indexes_no_gimp = [(i not in df_gimp.RGIId.values) for i in rgidf.RGIId]
keep_gimp = [(i in df_gimp.RGIId.values) for i in rgidf.RGIId]
rgidf_gimp = rgidf.iloc[keep_gimp]

rgidf = rgidf.iloc[keep_indexes_no_gimp]

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))
log.info('Number of glaciers with GIMP: {}'.format(len(rgidf_gimp)))

# Go - initialize working directories
# -----------------------------------
if run_mode:
    keep_index_to_run = [(i in config['RGI_id_to_test']) for i in rgidf.RGIId]
    rgidf = rgidf.iloc[keep_index_to_run]
    log.info('Starting run for RGI reg: ' + rgi_region)
    log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))
    gdirs = workflow.init_glacier_directories(rgidf)
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                 source='ARCTICDEM')
else:
    gdirs = workflow.init_glacier_directories(rgidf)
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                 source='ARCTICDEM')

    gdirs_gimp = workflow.init_glacier_directories(rgidf_gimp)
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs_gimp,
                                 source='GIMP')
    gdirs.extend(gdirs_gimp)

# Pre-pro tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM preprocessing finished! Time needed: %02d:%02d:%02d" %
         (h, m, s))

ids = []
vel_fls_avg = []
err_fls_avg = []
vel_calving_front = []
err_calving_front = []
rel_tol_fls = []
rel_tol_calving_front = []
length_fls = []

files_no_data = []

ds = utils_h.open_thick_raster(os.path.join(input_data_path,
                                              config['h_file_path']))
dr = utils_h.open_thick_raster(os.path.join(input_data_path,
                                              config['h_error_file_path']))

data_frame = []
rgi_ids = []
thick_end = []
error_end = []

for gdir in gdirs:

    # first we compute the centerlines as shapefile to crop the satellite
    # data
    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    ds_fls, dr_fls = utils_h.crop_thick_data_to_flowline(ds, dr, shp)

    thick, error, lon, lat = utils_h.calculate_observation_thickness(gdir,
                                                                     ds_fls,
                                                                     dr_fls,
                                                                     return_profile=True)

    d = {'H_flowline': thick,
         'H_flowline_error': error,
         'lon': lon,
         'lat': lat
         }
    data_frame = pd.DataFrame(data=d)
    data_frame.to_csv(os.path.join(cfg.PATHS['working_dir'], gdir.rgi_id + '.csv'))

    thick_f, error_f = utils_h.calculate_observation_thickness(gdir,
                                                               ds_fls,
                                                               dr_fls)

    rgi_ids = np.append(rgi_ids, gdir.rgi_id)
    thick_end = np.append(thick_end, thick_f)
    error_end = np.append(error_end, error_f)

dr = {'RGI_ID': rgi_ids,
      'thick_end': thick_end,
      'error_end': error_end,
     }

df_r = pd.DataFrame(data=dr)
df_r.to_csv(cfg.PATHS['working_dir'] + '/thickness_observations.csv')
