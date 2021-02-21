# This will run OGGM and obtain surface mass balance means from RACMO
# over a reference period
# time_start = '2015-10-01'
# time_end = '2018-09-31'
# This will be use to calibrate the k parameter in Greenland
from __future__ import division
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from configobj import ConfigObj
import time
import salem

# Imports oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils

# Module logger
import logging
log = logging.getLogger(__name__)
# Time
start = time.time()

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# misc and racmo modules
from k_tools import utils_racmo as utils_racmo

# Region Greenland
rgi_region = '05'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()
rgi_version = '61'

SLURM_WORKDIR = os.environ["WORKDIR"]

# Local paths (where to write output and where to download input)
WORKING_DIR = SLURM_WORKDIR
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True

# We make the border 20 so we can use the Columbia itmix DEM
cfg.PARAMS['border'] = 20
# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['min_mu_star'] = 0.0
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False
cfg.PARAMS['free_board_marine_terminating'] = 10, 150

# RACMO data path
racmo_path = os.path.join(MAIN_PATH, config['racmo_path'])

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# Exclude glaciers with prepro erros
de = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

# We use intersects
cfg.set_intersects_db(os.path.join(MAIN_PATH, config['intercepts']))

rgidf = rgidf.sort_values('RGIId', ascending=True)

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

# # Run a single id for testing
# glacier = ['RGI60-05.00304', 'RGI60-05.08443']
# keep_indexes = [(i in glacier) for i in rgidf.RGIId]
# rgidf = rgidf.iloc[keep_indexes]

# Remove glaciers that need to be model with gimp
df_gimp = pd.read_csv(os.path.join(MAIN_PATH, config['glaciers_gimp']))
keep_indexes_no_gimp = [(i not in df_gimp.RGIId.values) for i in rgidf.RGIId]
keep_gimp = [(i in df_gimp.RGIId.values) for i in rgidf.RGIId]
rgidf_gimp = rgidf.iloc[keep_gimp]

rgidf = rgidf.iloc[keep_indexes_no_gimp]

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))
log.info('Number of glaciers with GIMP: {}'.format(len(rgidf_gimp)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_directories(rgidf)

workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                             source='ARCTICDEM')

gdirs_gimp = workflow.init_glacier_directories(rgidf_gimp)
workflow.execute_entity_task(tasks.define_glacier_region, gdirs_gimp,
                             source='GIMP')

gdirs.extend(gdirs_gimp)

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

time_start = '2015-10-01'
time_end = '2018-10-01'
alias = 'AS-Oct'

for gdir in gdirs:

    utils_racmo.process_racmo_data(gdir,
                                   racmo_path,
                                   time_start=time_start,
                                   time_end=time_end,
                                   alias=alias)

    # We compute a calving flux from RACMO data
    out = utils_racmo.get_smb31_from_glacier(gdir)

    if out['smb_mean'] is None:
        print('There is no RACMO data for this glacier')
        files_no_data = np.append(files_no_data, gdir.rgi_id)
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

d = {'RGIId': files_no_data}
df = pd.DataFrame(data=d)

df.to_csv(cfg.PATHS['working_dir'] + 'glaciers_with_no_racmo_data.csv')

dr = {'RGI_ID': ids,
      'smb_mean': smb_avg,
      'smb_std': smb_std,
      'smb_cum': smb_cum,
      'q_calving_RACMO_mean': racmo_calving_avg,
      'q_calving_RACMO_mean_std': racmo_calving_avg_std,
      'q_calving_RACMO_cum': racmo_calving_cum}

df_r = pd.DataFrame(data=dr)
df_r.to_csv(cfg.PATHS['working_dir']+'racmo_data_'+time_start+time_end+'_.csv')
