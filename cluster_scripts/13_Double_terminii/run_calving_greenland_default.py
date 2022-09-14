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

# Get glaciers that have double terminii to run
df = pd.read_csv(os.path.join(input_data_path,
                              config['glaciers_double_terminii']))
ids = df.RGIId.values
keep_errors = [(i in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

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

# Climate tasks -- we make sure that calving is = 0 for all tidewater
for gdir in gdirs:
    gdir.inversion_calving_rate = 0

workflow.climate_tasks(gdirs, base_url=config['climate_url'])

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs)

workflow.execute_entity_task(its_live.velocity_to_gdir, gdirs, add_error=True)


for gdir in gdirs:
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()

    # get the wind data at 10000 m a.s.l.
    u = ds.obs_icevel_x.where(ds.glacier_mask == 1)
    v = ds.obs_icevel_y.where(ds.glacier_mask == 1)
    ws = (u ** 2 + v ** 2) ** 0.5

    min_vel = np.min(ws)
    med_vel = np.median(ws)
    mean = np.round(np.mean(ws),2)
    max_vel = np.round(np.max(ws), 2)

    # Quiver only every 3rd grid point
    us = u[1::3, 1::3]
    vs = v[1::3, 1::3]

    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    if gdir.rgi_id == 'RGI60-05.10315_d14':
        loc_pos = 'upper right'
    else:
        loc_pos = 'upper left'

    f, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 6), ncols=3, nrows=1,
                                    gridspec_kw = {'wspace':0.2, 'hspace':0})
    # Show/save figure as desired.
    llkw = {'interval': 1}

    graphics.plot_googlemap(gdir, ax=ax1)
    at = AnchoredText('a', prop=dict(size=16), frameon=True, loc=loc_pos)
    ax1.add_artist(at)

    graphics.plot_catchment_width(gdir, ax=ax2, title=gdir.rgi_id, corrected=True,
                                  lonlat_contours_kwargs=llkw,
                                  add_colorbar=False, add_scalebar=True)
    at = AnchoredText('b', prop=dict(size=16), frameon=True, loc=loc_pos)
    ax2.add_artist(at)

    smap = ds.salem.get_map(countries=False)
    smap.set_shapefile(gdir.read_shapefile('outlines'))
    smap.set_shapefile(shp, color='r', linewidth=1.5)
    smap.set_data(ws)
    smap.set_lonlat_contours(interval=1)
    smap.set_cmap('Blues')

    # transform their coordinates to the map reference system and plot the arrows
    xx, yy = smap.grid.transform(us.x.values, us.y.values, crs=gdir.grid.proj)
    xx, yy = np.meshgrid(xx, yy)
    qu = ax3.quiver(xx, yy, us.values, vs.values)
    qk = ax3.quiverkey(qu, min_vel, med_vel, max_vel, str(max_vel)+'m s$^{-1}$',
                      labelpos='E', coordinates='axes')
    smap.visualize(ax=ax3, cbar_title='ITSlive velocity \n [m/yr]',
                   title=gdir.rgi_id)

    at = AnchoredText('c', prop=dict(size=16), frameon=True, loc=loc_pos)
    ax3.add_artist(at)

    plt.savefig(os.path.join(cfg.PATHS['working_dir'],
                             gdir.rgi_id+'.png'),
                bbox_inches='tight')
    plt.clf()
    plt.close(f)

    # f, ax = plt.subplots(figsize=(9, 9))
    # smap = ds.salem.get_map(countries=False)
    # smap.set_shapefile(gdir.read_shapefile('outlines'))
    # smap.set_shapefile(shp, color='r', linewidth=1.5)
    # smap.set_data(ws)
    # smap.set_cmap('Blues')
    # smap.visualize(ax=ax, cbar_title='ITSlive velocity \n [m/yr]', title=gdir.rgi_id)
    # #smap.append_colorbar(ax=ax)
    # plt.savefig(os.path.join(cfg.PATHS['working_dir'],
    #                          gdir.rgi_id + '_velocity.png'),
    #             bbox_inches='tight')
    # plt.clf()
    # plt.close(f)

# Compile output
df_stats = misc.compile_exp_statistics(gdirs)

filesuffix = '_non_calving_glaciers_'
df_stats.to_csv(os.path.join(cfg.PATHS['working_dir'],
                             ('glacier_statistics' + filesuffix + '.csv')))

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %02d:%02d:%02d" % (h, m, s))