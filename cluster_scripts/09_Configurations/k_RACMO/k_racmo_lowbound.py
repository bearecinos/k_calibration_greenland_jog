# This will run k x factors experiment only for MT
# and compute model surface velocity and frontal ablation fluxes
# per TW glacier in Greenland
from __future__ import division
import os
import sys
import geopandas as gpd
import salem
import pandas as pd
import numpy as np
from configobj import ConfigObj
import time
import argparse

# Imports oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm.core import inversion

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

from k_tools import misc
from k_tools import utils_velocity as utils_vel

# Region Greenland
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
    SLURM_WORKDIR = os.environ.get("OUTDIR_low")
    # Local paths (where to write output and where to download input)
    WORKING_DIR = SLURM_WORKDIR
    cfg.PATHS['working_dir'] = WORKING_DIR

print(cfg.PATHS['working_dir'])

# Use multiprocessing
if run_mode:
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 5
else:
    # ONLY IN THE CLUSTER!
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 6

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

# Reads racmo calibration output
output_racmo = os.path.join(MAIN_PATH, config['racmo_calibration_results'])

no_solution = os.path.join(output_racmo, 'glaciers_with_no_solution.csv')
d_no_sol = pd.read_csv(no_solution)
ids_rgi = d_no_sol.RGIId.values
keep_no_solution = [(i not in ids_rgi) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_solution]

no_racmo_data = os.path.join(output_racmo, 'glaciers_with_no_racmo_data.csv')
d_no_data = pd.read_csv(no_racmo_data)
ids_no_data = d_no_data.RGIId.values
keep_no_data = [(i not in ids_no_data) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_data]

# Read calibration results
calibration_results_racmo = os.path.join(MAIN_PATH,
                                         config['linear_fit_to_data'])

path_to_file = os.path.join(calibration_results_racmo,
                            'racmo_fit_calibration_results.csv')

dc = pd.read_csv(path_to_file, index_col='RGIId')

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

# Climate tasks -- we make sure that calving is = 0 for all tidewater
for gdir in gdirs:
    gdir.inversion_calving_rate = 0

workflow.climate_tasks(gdirs, base_url=config['climate_url'])

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs)

# Compile climate statistics
utils.compile_climate_statistics(gdirs)

# Find out which glaciers have negative temperatures
RGI_ids = []
month_count = []
PDM_temp_free_board = []
prcp_at_the_top = []

for gdir in gdirs:

    PDM_temp, month_num, prcp = misc.calculate_pdm(gdir)

    RGI_ids = np.append(RGI_ids, gdir.rgi_id)
    month_count = np.append(month_count, month_num)
    PDM_temp_free_board = np.append(PDM_temp_free_board, PDM_temp)
    prcp_at_the_top = np.append(prcp_at_the_top, prcp)


d_climate = {'rgi_id': RGI_ids,
             'Number_PDM': month_count,
             'temp_PDM': PDM_temp_free_board,
             'total_prcp_top': prcp_at_the_top}

df_clima = pd.DataFrame(data=d_climate).set_index('rgi_id')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM without calving plus stats is done! "
         "Time needed: %02d:%02d:%02d" %
         (h, m, s))

cross = []
surface = []
flux = []
mu_star = []
k_used = []
ids = []

# Compute a calving flux
for gdir in gdirs:
    sel = dc[dc.index == gdir.rgi_id]
    k_value = sel.k_for_lw_bound.values

    cfg.PARAMS['continue_on_error'] = False
    cfg.PARAMS['inversion_calving_k'] = float(k_value)
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['use_kcalving_for_ru'] = True

    out = inversion.find_inversion_calving(gdir)

    inversion.compute_velocities(gdir)

    vel_out = utils_vel.calculate_model_vel(gdir)

    vel_surface = vel_out[2]
    vel_cross = vel_out[3]

    cross = np.append(cross, vel_cross)
    surface = np.append(surface, vel_surface)
    k_used = np.append(k_used, k_value)
    ids = np.append(ids, gdir.rgi_id)

d_vel = {'rgi_id': ids,
         'k_value': k_used,
         'velocity_cross': cross,
         'velocity_surf': surface}

df_vel = pd.DataFrame(data=d_vel).set_index('rgi_id')
exp_name = 'k_racmo_lowbound_'
df_vel.columns = exp_name + df_vel.columns

df_stats = misc.compile_exp_statistics(gdirs)

df_core = misc.get_core_data(df_stats)
df_exp = misc.summarize_exp(df_stats, exp_name=exp_name)

df_stats_final = pd.concat([df_core, df_exp, df_clima, dc, df_vel], axis=1)

df_stats_final.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                   ('glacier_statistics_calving_' +
                                    exp_name + '.csv')))

misc.reset_per_glacier_working_dir()