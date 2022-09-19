# This will run OGGM and obtain surface mass balance means from RACMO
# over a reference period 1961-1990
# This will be use to calibrate the k parameter in Greenland
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
from oggm import workflow, utils
from oggm import tasks
from oggm.workflow import execute_entity_task

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

# misc and racmo modules
from k_tools import utils_racmo as utils_racmo
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
    cfg.PATHS['working_dir'] = utils.get_temp_dir('GP-test-run')
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

# RACMO data path
racmo_path = os.path.join(input_data_path, config['racmo_path'])

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

execute_entity_task(tasks.glacier_masks, gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM dirs finished! Time needed: %02d:%02d:%02d" %
         (h, m, s))

ids = []
smb_avg = []
smb_std = []
smb_cum = []
racmo_calving_avg = []
racmo_calving_avg_std = []
racmo_calving_cum = []

files_no_data = []
area_no_data = []

time_start = '1961-01-01'
time_end = '1990-12-31'
alias = 'AS'

workflow.execute_entity_task(utils_racmo.process_racmo_data,
                             gdirs,
                             racmo_path=racmo_path,
                             time_start=time_start,
                             time_end=time_end,
                             alias=alias)

for gdir in gdirs:
    # We compute a calving flux from RACMO data
    out = utils_racmo.get_smb31_from_glacier(gdir)

    if out['smb_mean'] is None:
        print('There is no RACMO data for this glacier')
        files_no_data = np.append(files_no_data, gdir.rgi_id)
        area_no_data = np.append(area_no_data, gdir.rgi_area_km2)
    else:
        # We append everything
        ids = np.append(ids, gdir.rgi_id)
        smb_avg = np.append(smb_avg, out['smb_mean'])
        smb_std = np.append(smb_std, out['smb_std'])
        smb_cum = np.append(smb_cum, out['smb_cum'])
        racmo_calving_avg = np.append(racmo_calving_avg,
                                      out['smb_calving_mean'])
        racmo_calving_avg_std = np.append(racmo_calving_avg_std,
                                          out['smb_calving_std'])
        racmo_calving_cum = np.append(racmo_calving_cum,
                                      out['smb_calving_cum'])

d = {'RGIId': files_no_data,
     'Area (km)': area_no_data}
df = pd.DataFrame(data=d)

df.to_csv(os.path.join(cfg.PATHS['working_dir'], 'glaciers_with_no_racmo_data.csv'))

dr = {'RGI_ID': ids,
      'smb_mean': smb_avg,
      'smb_std': smb_std,
      'smb_cum': smb_cum,
      'q_calving_RACMO_mean': racmo_calving_avg,
      'q_calving_RACMO_mean_std': racmo_calving_avg_std,
      'q_calving_RACMO_cum': racmo_calving_cum}

df_r = pd.DataFrame(data=dr)
df_r.to_csv(os.path.join(cfg.PATHS['working_dir'], 'racmo_data_'+time_start+'_'+time_end+'.csv'))

misc.reset_per_glacier_working_dir()