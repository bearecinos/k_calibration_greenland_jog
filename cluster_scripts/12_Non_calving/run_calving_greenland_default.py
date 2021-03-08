# This will run OGGM preprocessing task and the inversion with calving
# For Greenland with default MB calibration and DEM: Glims
from __future__ import division
import logging
import os
import geopandas as gpd
import salem
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import sys
from configobj import ConfigObj
# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics
from oggm.shop import its_live
# Time
import time
start = time.time()
# Module logger
log = logging.getLogger(__name__)

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

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
sys.path.append(MAIN_PATH)
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

from k_tools import misc

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['border'] = 20
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

rgidf = rgidf.sort_values('RGIId', ascending=True)

# Read Areas for the ice-cap computed in OGGM during
# the pre-processing runs
df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))
df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

# Assign an area to the ice cap from OGGM to avoid errors
rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
          'Area'] = df_prepro_ic.rgi_area_km2.values

# Remove Land-terminating
glac_type = ['0']
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Remove glaciers with strong connection to the ice sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Exclude glaciers with prepro erros
de = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

# Keep glaciers with no calving solution
# Reads racmo calibration output
output_racmo = os.path.join(MAIN_PATH,
                            config['racmo_calibration_results'])
path_no_solution = os.path.join(output_racmo, 'glaciers_with_no_solution.csv')

no_sol_ids = misc.read_rgi_ids_from_csv(path_no_solution)
keep_no_solution = [(i in no_sol_ids) for i in rgidf.RGIId]
rgidf_no_sol = rgidf.iloc[keep_no_solution]

# Run a single id for testing
glacier = ['RGI60-05.10315_d11', 'RGI60-05.10315_d14']
keep_indexes = [(i in glacier) for i in rgidf.RGIId]
rgidf_test = rgidf.iloc[keep_indexes]

# # Remove glaciers that need to be model with gimp
# df_gimp = pd.read_csv(os.path.join(MAIN_PATH, config['glaciers_gimp']))
# keep_indexes_no_gimp = [(i not in df_gimp.RGIId.values) for i in rgidf.RGIId]
# keep_gimp = [(i in df_gimp.RGIId.values) for i in rgidf.RGIId]
# rgidf_gimp = rgidf.iloc[keep_gimp]
#
# rgidf = rgidf.iloc[keep_indexes_no_gimp]
#
log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers with no solution: {}'.format(len(rgidf_no_sol)))
log.info('Number of glaciers Ice cap: {}'.format(len(rgidf_test)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_directories(rgidf_no_sol)

workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                             source='ARCTICDEM')

gdirs_ice_cap = workflow.init_glacier_directories(rgidf_test)
workflow.execute_entity_task(tasks.define_glacier_region, gdirs_ice_cap,
                             source='ARCTICDEM')
#
gdirs.extend(gdirs_ice_cap)

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
