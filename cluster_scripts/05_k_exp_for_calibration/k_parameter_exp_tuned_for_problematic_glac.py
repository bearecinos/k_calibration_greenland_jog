# This will run k x factors experiment only for MT
# and compute model surface velocity and frontal ablation fluxes
# per TW glacier in Greenland
from __future__ import division
import os
import sys
import numpy as np
import geopandas as gpd
import salem
import pandas as pd
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

# velocity module
from k_tools import utils_velocity as utils_vel
from k_tools import misc

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

# Repeat experiment tuned for problematic glaciers.
dp = pd.read_csv(os.path.join(MAIN_PATH, config['problematic_glaciers']))
ids_p = dp.RGIId.values
keep_problematic = [(i in ids_p) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_problematic]


log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers with ArcticDEM: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_directories(rgidf)

workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                             source='ARCTICDEM')

# Prepro tasks
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

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM without calving is done! Time needed: %02d:%02d:%02d" %
         (h, m, s))

cfg.PARAMS['use_kcalving_for_inversion'] = True
cfg.PARAMS['use_kcalving_for_ru'] = True

k_factors = np.arange(0.01, 0.1, 0.0001)

for gdir in gdirs:
    cross = []
    surface = []
    flux = []
    mu_star = []
    k_used = []

    for k in k_factors:

        print('Calculating loop for ', gdir.rgi_id)

        # Find a calving flux.
        cfg.PARAMS['inversion_calving_k'] = k
        out = inversion.find_inversion_calving(gdir)
        if out is None:
            continue

        calving_flux = out['calving_flux']
        calving_mu_star = out['calving_mu_star']

        inversion.compute_velocities(gdir)

        vel_out = utils_vel.calculate_model_vel(gdir)

        vel_surface = vel_out[2]
        vel_cross = vel_out[3]

        cross = np.append(cross, vel_cross)
        surface = np.append(surface, vel_surface)
        flux = np.append(flux, calving_flux)
        mu_star = np.append(mu_star, calving_mu_star)
        k_used = np.append(k_used, k)

        if mu_star[-1] == 0:
            break

    d = {'k_values': k_used,
         'velocity_cross': cross,
         'velocity_surf': surface,
         'calving_flux': flux,
         'mu_star': mu_star}

    df = pd.DataFrame(data=d)

    df.to_csv(os.path.join(cfg.PATHS['working_dir'], gdir.rgi_id + '.csv'))