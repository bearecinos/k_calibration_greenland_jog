# This will run OGGM and obtain thickness data from Millan et al. 2022
# It will give you thickness data at the last pixel of the flowline
# and along the main centerline, data its store as a .csv file
# for each glacier in the main working directory

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
from oggm.shop import millan22

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
from k_tools import utils_thick as utils_h
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

# Remove Land-terminating
glac_type = ['0']
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Remove glaciers with strong connection to the ice sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# # Keep only problematic glacier from marcos list
# path_to_problematic = os.path.join(input_data_path,
#                                    'millan_problematic/class2_ids.txt')
# dl = pd.read_csv(path_to_problematic)
# ids_l = dl.rgi_id.values
# keep_problem = [(i in ids_l) for i in rgidf.RGIId]
# rgidf = rgidf.iloc[keep_problem]

# Exclude glaciers with prepro-erros
de = pd.read_csv(os.path.join(input_data_path, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

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
    keep_index_to_run = [(i in ['RGI60-05.12047', 'RGI60-05.00800']) for i in rgidf.RGIId]
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

if correct_width:
    for gdir in gdirs:
        if gdir.rgi_id in dfmac.index:
            width = dfmac.loc[gdir.rgi_id]['gate_length_km']
            tasks.terminus_width_correction(gdir, new_width=width*1000)

workflow.execute_entity_task(millan22.thickness_to_gdir, gdirs, add_error=True)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM preprocessing finished! Time needed: %02d:%02d:%02d" %
         (h, m, s))

path_to_output = cfg.PATHS['working_dir']+'/'+ 'millan_22'
if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)

ids_no_data = []
area_no_data = []
rgi_ids = []
thick_end = []
error_end = []

for gdir in gdirs:

    # first we compute the centerlines as shapefile to crop the satellite
    # data
    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    ds = xr.open_dataset(gdir.get_filepath('gridded_data'))

    try:
        ds.millan_ice_thickness.attrs['pyproj_srs'] = ds.attrs['pyproj_srs']
        ds.millan_ice_thickness_err.attrs['pyproj_srs'] = ds.attrs['pyproj_srs']
    except AttributeError:
        print('There is no data for this glacier', gdir.rgi_id)
        ids_no_data = np.append(ids_no_data, gdir.rgi_id)
        area_no_data = np.append(area_no_data, gdir.rgi_area_km2)
        continue

    ds_fls, dr_fls = utils_h.crop_thick_data_to_flowline(ds.millan_ice_thickness,
                                                         ds.millan_ice_thickness_err,
                                                         shp)

    H_fls, H_err_fls, lon, lat = utils_h.calculate_observation_thickness(gdir, ds_fls, dr_fls)

    # Saving the individual flowline data
    d = {'H_flowline': H_fls,
         'H_flowline_error': H_err_fls,
         'lon': lon,
         'lat': lat
         }
    data_frame = pd.DataFrame(data=d)
    data_frame.to_csv(os.path.join(path_to_output, gdir.rgi_id + '.csv'))

    rgi_ids = np.append(rgi_ids, gdir.rgi_id)
    thick_end = np.append(thick_end, H_fls[-1])
    error_end = np.append(error_end, H_err_fls[-1])

# Saving the end of the flowline data
dr = {'RGI_ID': rgi_ids,
      'thick_end': thick_end,
      'error_end': error_end,
     }

df_r = pd.DataFrame(data=dr)
df_r.to_csv(os.path.join(path_to_output,'thickness_observations.csv'))

dnodata = {'RGIId': ids_no_data,
           'Area (km)': area_no_data}
df_nodata = pd.DataFrame(data=dnodata)
df_nodata.to_csv(cfg.PATHS['working_dir'] + '/glaciers_with_no_thickness_data.csv')

misc.reset_per_glacier_working_dir()
