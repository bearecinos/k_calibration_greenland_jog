# This will run OGGM preprocessing task and the inversion with calving
# For Greenland with default MB calibration and DEM: GLIMS and ArcticDEM
from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import sys
import geopandas as gpd
import salem
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
from configobj import ConfigObj
import argparse

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm import graphics
from oggm.core import inversion
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
# Import our own module
from k_tools import misc
from k_tools import utils_velocity as utils_vel

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
    cfg.PATHS['working_dir'] = utils.get_temp_dir('GP-test-run')
else:
    SLURM_WORKDIR = os.environ["OUTDIR"]
    # Local paths (where to write output and where to download input)
    WORKING_DIR = SLURM_WORKDIR
    cfg.PATHS['working_dir'] = WORKING_DIR


print(cfg.PATHS['working_dir'])

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

# Get glaciers that have no solution
output_itslive = os.path.join(MAIN_PATH,
                              config['vel_calibration_results_itslive'])

no_solution = os.path.join(output_itslive, 'glaciers_with_no_solution.csv')
d_no_sol = pd.read_csv(no_solution)
ids_rgi = d_no_sol.RGIId.values
keep_no_solution = [(i in ids_rgi) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_solution]

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))

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


print(gdirs)
#
# execute_entity_task(tasks.glacier_masks, gdirs)

# Calculate the Pre-processing tasks
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


# Climate tasks -- we make sure that calving is = 0 for all tidewater
for gdir in gdirs:
    gdir.inversion_calving_rate = 0

workflow.climate_tasks(gdirs, base_url=config['climate_url'])

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs)

workflow.execute_entity_task(its_live.velocity_to_gdir, gdirs, add_error=True)

ids_with_vel = []
vel_calving_front = []
err_calving_front = []
area_vel = []

ids_with_low_vel = []
vel_calving_front_low = []
err_calving_front_low = []
area_low_vel = []

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
    dvel = np.sqrt(vx ** 2 + vy ** 2)

    dvel.attrs['pyproj_srs'] = proj

    ex = file_vel.err_icevel_x
    ey = file_vel.err_icevel_y
    derr = np.sqrt(ex ** 2 + ey ** 2)
    derr.attrs['pyproj_srs'] = proj

    # we crop the satellite data to the centerline shape file
    dvel_fls, derr_fls = utils_vel.crop_vel_data_to_flowline(dvel, derr, shp)

    out = utils_vel.calculate_itslive_vel(gdir, dvel_fls, derr_fls)

    if out[2] > 10:
        ids_with_vel = np.append(ids_with_vel, gdir.rgi_id)
        area_vel = np.append(area_vel, gdir.rgi_area_km2)
        vel_calving_front = np.append(vel_calving_front, out[2])
        err_calving_front = np.append(err_calving_front, out[3])
    else:
        ids_with_low_vel = np.append(ids_with_low_vel, gdir.rgi_id)
        area_low_vel = np.append(area_low_vel, gdir.rgi_area_km2)
        vel_calving_front_low = np.append(vel_calving_front_low, out[2])
        err_calving_front_low = np.append(err_calving_front_low, out[3])

d = {'RGIId': ids_with_low_vel,
     'Area (km)': area_low_vel,
     'Velocity calving front': vel_calving_front_low,
     'Error velocity calving front': err_calving_front_low}

df = pd.DataFrame(data=d)
df.to_csv(cfg.PATHS['working_dir'] + '/glaciers_with_low_velocity_data.csv')

dr = {'RGIId': ids_with_vel,
     'Area (km)': area_vel,
     'Velocity calving front': vel_calving_front,
     'Error velocity calving front': err_calving_front}

df_r = pd.DataFrame(data=dr)
df_r.to_csv(cfg.PATHS['working_dir'] + '/glaciers_that_should_calve.csv')

# Compile output
df_stats = misc.compile_exp_statistics(gdirs)

filesuffix = '_non_calving_glaciers_'
df_stats.to_csv(os.path.join(cfg.PATHS['working_dir'],
                             ('glacier_statistics' + filesuffix + '.csv')))

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %02d:%02d:%02d" % (h, m, s))
