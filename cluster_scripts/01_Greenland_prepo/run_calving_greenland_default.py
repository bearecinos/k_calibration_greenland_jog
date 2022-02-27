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
from oggm.core import inversion

# Time
import time
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
# Import our own module
from k_tools import misc

# Regions:
# Greenland
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

# Get glaciers that belong to the ice cap.
rgidf_ice_cap = rgidf[rgidf['RGIId'].str.match('RGI60-05.10315')]
# Get the id's for filter
ice_cap_ids = rgidf_ice_cap.RGIId.values

# Removing the Ice cap
keep_indexes = [(i not in ice_cap_ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_indexes]

# Remove Land-terminating
glac_type = ['0']
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Remove glaciers with strong connection to the ice sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Remove glaciers that need to be model with gimp
df_gimp = pd.read_csv(os.path.join(input_data_path, config['glaciers_gimp']))
keep_indexes_no_gimp = [(i not in df_gimp.RGIId.values) for i in rgidf.RGIId]
keep_gimp = [(i in df_gimp.RGIId.values) for i in rgidf.RGIId]
rgidf_gimp = rgidf.iloc[keep_gimp]

rgidf = rgidf.iloc[keep_indexes_no_gimp]

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

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

#print(gdirs)
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

# Climate tasks -- we make sure that calving is = 0 for all tidewater
for gdir in gdirs:
    gdir.inversion_calving_rate = 0

workflow.climate_tasks(gdirs, base_url=config['climate_url'])

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs)

df_stats = misc.compile_exp_statistics(gdirs)

filesuffix = '_greenland_no_calving_with_sliding_'

df_stats.to_csv(os.path.join(cfg.PATHS['working_dir'],
                             ('glacier_statistics' + filesuffix + '.csv')))

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %02d:%02d:%02d" % (h, m, s))

cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['use_kcalving_for_inversion'] = True
cfg.PARAMS['use_kcalving_for_ru'] = True

glac_errors = []
glac_dont_calve = []

# Compute a calving flux
for gdir in gdirs:
    try:
        out = inversion.find_inversion_calving(gdir)
    except:
        print('there was an error in calving', gdir.rgi_id)
        glac_errors = np.append(glac_errors, gdir.rgi_id)
        pass
    if out is None:
        glac_dont_calve = np.append(glac_dont_calve, gdir.rgi_id)
        pass

d = {'RGIId': glac_errors}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(cfg.PATHS['working_dir'], 'glaciers_with_prepro_errors'+'.csv'))

s = {'RGIId': glac_dont_calve}
ds = pd.DataFrame(data=s)
ds.to_csv(os.path.join(cfg.PATHS['working_dir'],
                       'glaciers_dont_calve_with_cgf_params'+'.csv'))

cfg.PARAMS['continue_on_error'] = True

df_stats_c = misc.compile_exp_statistics(gdirs)

filesuffix_c = '_greenland_calving_with_sliding'

df_stats_c.to_csv(os.path.join(cfg.PATHS['working_dir'],
                               ('glacier_statistics' + filesuffix_c + '.csv')))

misc.reset_per_glacier_working_dir()
