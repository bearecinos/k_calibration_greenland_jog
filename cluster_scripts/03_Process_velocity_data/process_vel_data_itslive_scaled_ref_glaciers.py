# This will run OGGM and extract velocity data from ITSlive at 120 m
# It will give you velocity averages along the last one third of the
# main flow-line with the respective uncertainty from the ITSLIVE data.
# Python imports
from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import sys
import geopandas as gpd
import salem
import numpy as np
import pandas as pd
from configobj import ConfigObj
import argparse
import math
import xarray as xr

# Imports oggm
import oggm.cfg as cfg
from oggm import workflow, utils
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm.shop import its_live

# Time
import time
start = time.time()

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-mode", type=bool, default=False, help="pass running mode")
parser.add_argument("-correct_width", type=bool, default=False, help="correct terminus width with extra data")
args = parser.parse_args()
config_file = args.conf
run_mode = args.mode
correct_width = args.correct_width

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']
input_data_path = config['input_data_folder']
sys.path.append(MAIN_PATH)

# velocity module
from k_tools import utils_velocity as utils_vel
from k_tools import misc

# Regions:
# Greenland
rgi_region = '05'
rgi_version = '62'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------
cfg.initialize()

# Define working directories (either local if run_mode = true)
# or in the cluster environment
if run_mode:
    cfg.PATHS['working_dir'] = utils.get_temp_dir('GP-test-run-itslive')
else:
    SLURM_WORKDIR = os.environ.get("OUTDIR")
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
# We use width corrections
# From Will's flux gates
data_link = os.path.join(input_data_path,
                         'wills_data.csv')
dfmac = pd.read_csv(data_link, index_col=0)
dfmac = dfmac[dfmac.Region_name == 'Greenland']

if not run_mode:
    # Read Areas for the ice-cap computed in OGGM during
    # the pre-processing runs
    df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                            config['ice_cap_prepro']))

    df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

    # Assign an area to the ice cap from OGGM to avoid errors
    rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
              'Area'] = df_prepro_ic.rgi_area_km2.values

# Keep only glaciers for testing
path_to_test = os.path.join(input_data_path,
                                   'reference_glaciers_test.csv')
dl = pd.read_csv(path_to_test)
ids_l = dl.rgi_id.values
keep_problem = [(i in ids_l) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_problem]

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
if run_mode:
    log.info('Starting run for RGI reg: ' + rgi_region)
    log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))
    gdirs = workflow.init_glacier_directories(rgidf)
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                 source='ARCTICDEM')
else:
    gdirs = workflow.init_glacier_directories(rgidf)
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                 source='ARCTICDEM')

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

if correct_width:
    for gdir in gdirs:
        if gdir.rgi_id in dfmac.index:
            width = dfmac.loc[gdir.rgi_id]['gate_length_km']
            tasks.terminus_width_correction(gdir, new_width=width*1000)

workflow.execute_entity_task(its_live.velocity_to_gdir, gdirs, add_error=True)

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
area_no_data = []

for gdir in gdirs:

    # first we compute the centerlines as shapefile to crop the satellite
    # data
    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    file_vel = xr.open_dataset(gdir.get_filepath('gridded_data'))
    proj = file_vel.attrs['pyproj_srs']

    vx = file_vel.obs_icevel_x
    vy = file_vel.obs_icevel_y
    dvel = np.sqrt(vx**2 + vy**2)

    dvel.attrs['pyproj_srs'] = proj

    ex = file_vel.err_icevel_x
    ey = file_vel.err_icevel_y
    derr = np.sqrt(ex**2 + ey**2)
    derr.attrs['pyproj_srs'] = proj

    # we crop the satellite data to the centerline shape file
    dvel_fls, derr_fls = utils_vel.crop_vel_data_to_flowline(dvel, derr, shp)

    out = utils_vel.calculate_itslive_vel(gdir, dvel_fls, derr_fls)

    if math.isfinite(out[2]) and math.isfinite(out[0]):
        ids = np.append(ids, gdir.rgi_id)
        vel_fls_avg = np.append(vel_fls_avg, out[0])
        err_fls_avg = np.append(err_fls_avg, out[1])

        rel_tol_fls = np.append(rel_tol_fls, out[1] / out[0])
        vel_calving_front = np.append(vel_calving_front, out[2])
        err_calving_front = np.append(err_calving_front, out[3])
        rel_tol_calving_front = np.append(rel_tol_calving_front,
                                          out[3] / out[2])
        length_fls = np.append(length_fls, out[4])
    else:
        print('There is no velocity data for this glacier')
        files_no_data = np.append(files_no_data, gdir.rgi_id)
        area_no_data = np.append(area_no_data, gdir.rgi_area_km2)

d = {'RGIId': files_no_data,
     'Area (km)': area_no_data}
df = pd.DataFrame(data=d)

df.to_csv(cfg.PATHS['working_dir'] + '/glaciers_with_no_velocity_data.csv')

dr = {'RGI_ID': ids,
      'vel_fls': vel_fls_avg,
      'error_fls': err_fls_avg,
      'rel_tol_fls': rel_tol_fls,
      'vel_calving_front': vel_calving_front,
      'error_calving_front': err_calving_front,
      'rel_tol_calving_front': rel_tol_calving_front}

df_r = pd.DataFrame(data=dr)
df_r.to_csv(cfg.PATHS['working_dir'] + '/velocity_observations.csv')

misc.reset_per_glacier_working_dir()
