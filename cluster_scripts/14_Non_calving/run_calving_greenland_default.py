# This will run OGGM preprocessing task and the inversion with calving
# For Greenland with default MB calibration and DEM: Glims
from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import geopandas as gpd
import salem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from configobj import ConfigObj

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils, graphics
from oggm.core import inversion
from oggm.shop import rgitopo

# Time
import time
start = time.time()

# Regions:
# Greenland
rgi_region = '05'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()

rgi_version = '61'

SLURM_WORKDIR = os.environ["WORKDIR"]
# Local paths (where to write output and where to download input)
WORKING_DIR = SLURM_WORKDIR
cfg.PATHS['working_dir'] = WORKING_DIR

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
config = ConfigObj(os.path.join(MAIN_PATH,'config.ini'))
sys.path.append(MAIN_PATH)

from k_tools import misc

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

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# We use intersects
cfg.set_intersects_db(os.path.join(MAIN_PATH, config['intercepts']))

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

# Keep glaciers with no calving solution
# Reads racmo calibration output
output_racmo = os.path.join(MAIN_PATH,
                            config['racmo_calibration_results'])
path_no_solution = os.path.join(output_racmo,
                                      'glaciers_with_no_solution.csv')

no_sol_ids = misc.read_rgi_ids_from_csv(path_no_solution)
keep_no_solution = [(i in no_sol_ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_solution]

# Remove glaciers that need to be model with gimp
df_gimp = pd.read_csv(os.path.join(MAIN_PATH, config['glaciers_gimp']))
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
gdirs = workflow.init_glacier_directories(rgidf)

workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source='ARCTICDEM')

gdirs_gimp = workflow.init_glacier_directories(rgidf_gimp)
workflow.execute_entity_task(tasks.define_glacier_region, gdirs_gimp, source='GIMP')

gdirs.extend(gdirs_gimp)

# print(gdirs)
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

execute_entity_task(tasks.process_cru_data, gdirs)
execute_entity_task(tasks.local_t_star, gdirs)
execute_entity_task(tasks.mu_star_calibration, gdirs)

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs)

for gdir in gdirs:
    f = plt.figure()
    # Show/save figure as desired.
    llkw = {'interval': 1}
    graphics.plot_catchment_width(gdir, title=gdir.rgi_id, corrected=True,
                                  lonlat_contours_kwargs=llkw,
                                  add_colorbar=False, add_scalebar=False)
    plt.savefig(os.path.join(cfg.PATHS['working_dir'],
                             gdir.rgi_id+'.png'),
                bbox_inches='tight')
    plt.clf()
    plt.close(f)

# Compile output
utils.compile_glacier_statistics(gdirs,
                                 filesuffix='_non_calving_glaciers_')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %02d:%02d:%02d" % (h, m, s))