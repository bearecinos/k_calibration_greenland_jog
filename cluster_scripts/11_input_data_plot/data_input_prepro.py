import os
import sys
import salem
import xarray as xr
import numpy as np
import pyproj
from configobj import ConfigObj
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import geopandas as gpd
from oggm import cfg, workflow, graphics, tasks
from oggm.workflow import execute_entity_task
from oggm.core import inversion
from oggm.shop import its_live


MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import utils_velocity as utils_vel
from k_tools import utils_racmo as utils_racmo
from k_tools import misc as misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

# PARAMS for plots
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
sns.set_context('poster')

# The mask and geo reference data
mask_file = os.path.join(MAIN_PATH, config['mask_topo'])
ds_geo = xr.open_dataset(mask_file, decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs

# Getting hi-resolution coastline
coast_line = salem.read_shapefile(os.path.join(MAIN_PATH,
                                               config['coastline']))

# Velocity data paths
vel_itslive_path = os.path.join(MAIN_PATH, config['vel_golive'])
err_itslive_path = os.path.join(MAIN_PATH, config['error_vel_golive'])

vel_measures_path = os.path.join(MAIN_PATH, config['vel_path'])
err_measures_path = os.path.join(MAIN_PATH, config['error_vel_path'])

# Read vel data
dvel_itslive = utils_vel.open_vel_raster(vel_itslive_path)
derr_itslive = utils_vel.open_vel_raster(err_itslive_path)

dvel_measures = utils_vel.open_vel_raster(vel_measures_path)
derr_measures = utils_vel.open_vel_raster(err_measures_path)

# Paths to RACMO data
racmo_main_path = os.path.join(MAIN_PATH, config['racmo_path'])
smb_path = os.path.join(racmo_main_path,
                        'smb_rec.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

ds_smb = utils_racmo.open_racmo(smb_path, mask_file)
# .chunk({'time':20})
dsc = xr.open_dataset(smb_path, decode_times=False)
dsc.attrs['pyproj_srs'] = proj.srs

dsc['time'] = np.append(pd.period_range(start='2018.01.01',
                                        end='2018.12.01',
                                        freq='M').to_timestamp(),
                        pd.period_range(start='1958.01.01',
                                        end='2017.12.01',
                                        freq='M').to_timestamp())

ds_smb_two = dsc.isel(time=slice(48, 12*34))
# see = ds_smb_two.chunk({'time':2})
avg = ds_smb_two.SMB_rec.mean(dim='time').compute()

# OGGM run
cfg.initialize()
SLURM_WORKDIR = os.environ["WORKDIR"]
# Local paths (where to write output and where to download input)
WORKING_DIR = SLURM_WORKDIR
cfg.PATHS['working_dir'] = WORKING_DIR

# Find a glacier with good coverage of both data
gdirs = workflow.init_glacier_directories(['RGI60-05.00800'],
                                          from_prepro_level=3,
                                          prepro_border=10)
for gdir in gdirs:
    gdir.inversion_calving_rate = 0

execute_entity_task(tasks.process_cru_data, gdirs)
execute_entity_task(tasks.local_t_star, gdirs)
execute_entity_task(tasks.mu_star_calibration, gdirs)

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                    filesuffix='_without_calving_')

# Read calibration results
calibration_results_itslive = os.path.join(MAIN_PATH,
                                           config['linear_fit_to_data'])

path_to_file = os.path.join(calibration_results_itslive,
                            'velocity_fit_calibration_results_measures.csv')

dc = pd.read_csv(path_to_file, index_col='RGIId')

for gdir in gdirs:
    sel = dc[dc.index == gdir.rgi_id]
    k_value = sel.k_for_up_bound.values

    cfg.PARAMS['continue_on_error'] = False
    cfg.PARAMS['inversion_calving_k'] = float(k_value)
    out = inversion.find_inversion_calving(gdir)

gdir = gdirs[0]

misc.write_flowlines_to_shape(gdir, path=gdir.dir)
shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
shp = gpd.read_file(shp_path)

# Crop velocity from measures to raster
dvel_sel_m, derr_sel_m = utils_vel.crop_vel_data_to_glacier_grid(gdir,
                                                                 dvel_measures,
                                                                 derr_measures)

its_live.velocity_to_gdir(gdir)

with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
    # With mask
    # mask = ds.glacier_mask.data.astype(bool)
    # vx = ds.obs_icevel_x.where(mask)
    # vy = ds.obs_icevel_y.where(mask)
    # Without mask
    vx = ds.obs_icevel_x
    vy = ds.obs_icevel_y

vel_itslive = np.sqrt(vx**2 + vy**2)

# Crop and plot racmo data
ds_sel = utils_racmo.crop_racmo_to_glacier_grid(gdir, ds_smb)
# The time info is horrible
ds_sel['time'] = np.append(pd.period_range(start='2018.01.01',
                                           end='2018.12.01',
                                           freq='M').to_timestamp(),
                           pd.period_range(start='1958.01.01',
                                           end='2017.12.01',
                                           freq='M').to_timestamp())

# We select the time that we need 1960-1990
ds_smb_two_sel = ds_sel.isel(time=slice(48, 12*34))
# ds_smb_time_sel = ds_smb_two_sel.chunk({'time':2})

smb_avg_sel = ds_smb_two_sel.SMB_rec.mean(dim='time', skipna=True).compute()

print('Done calculating')

# Now plotting
fig1 = plt.figure(figsize=(16, 15), constrained_layout=True)

widths = [2, 2, 2]
heights = [2, 4, 2]

spec = gridspec.GridSpec(3, 3, wspace=1.0, hspace=0.05, width_ratios=widths,
                         height_ratios=heights)

# plot a
llkw = {'interval': 1}
ax0 = plt.subplot(spec[0])
graphics.plot_centerlines(gdir, ax=ax0, title='', add_colorbar=True,
                          lonlat_contours_kwargs=llkw, add_scalebar=False)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
graphics.plot_catchment_width(gdir, ax=ax1, title='', corrected=True,
                              lonlat_contours_kwargs=llkw,
                              add_colorbar=False, add_scalebar=False)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
misc.plot_inversion_diff(gdirs, ax=ax2, title='', linewidth=2,
                         add_colorbar=True,
                         lonlat_contours_kwargs=llkw,
                         add_scalebar=False)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
sm = dvel_measures.salem.get_map(countries=False)
sm.set_shapefile(oceans=True)
sm.set_data(dvel_measures.data)
sm.set_cmap('viridis')
sm.set_vmin(0)
sm.set_vmax(200)
x_conect, y_conect = sm.grid.transform(gdir.cenlon, gdir.cenlat)
ax3.scatter(x_conect, y_conect, s=80, marker="o", color='red')
ax3.text(x_conect, y_conect, s=gdir.rgi_id, color=sns.xkcd_rgb["white"],
         weight='black', fontsize=12)
sm.visualize(ax=ax3, cbar_title='MEaSUREs velocity \n [m/yr]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])
sm = dvel_itslive.salem.get_map(countries=False)
sm.set_shapefile(oceans=True)
sm.set_data(dvel_itslive.data)
sm.set_cmap('viridis')
sm.set_vmin(0)
sm.set_vmax(200)
x_conect, y_conect = sm.grid.transform(gdir.cenlon, gdir.cenlat)
ax4.scatter(x_conect, y_conect, s=80, marker="o", color='red')
ax4.text(x_conect, y_conect, s=gdir.rgi_id,
         color=sns.xkcd_rgb["white"],
         weight='black', fontsize=12)
sm.visualize(ax=ax4, cbar_title='ITSlive velocity \n [m/yr]')
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
sm = ds_smb.salem.get_map(countries=False)
sm.set_shapefile(oceans=True)
sm.set_data(avg)
sm.set_cmap('RdBu')
x_conect, y_conect = sm.grid.transform(gdir.cenlon, gdir.cenlat)
ax5.scatter(x_conect, y_conect, s=80, marker="o", color='red')
ax5.text(x_conect, y_conect, s=gdir.rgi_id,
         color=sns.xkcd_rgb["black"],
         weight='black', fontsize=10)
# sm.set_scale_bar(location=(0.78, 0.04))
sm.set_vmin(-200)
sm.set_vmax(200)
sm.visualize(ax=ax5, cbar_title='SMB mean 61-90 \n [mm. w.e]')
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc='upper left')
ax5.add_artist(at)

ax6 = plt.subplot(spec[6])
sm = dvel_sel_m.salem.get_map(countries=False)
sm.set_shapefile(gdir.read_shapefile('outlines'))
sm.set_shapefile(shp, color='r', linewidth=1.5)
sm.set_data(dvel_sel_m.data)
sm.set_cmap('viridis')
sm.set_vmin(0)
sm.set_vmax(200)
sm.set_lonlat_contours(interval=1)
sm.visualize(ax=ax6, addcbar=False)
at = AnchoredText('g', prop=dict(size=18), frameon=True, loc='upper left')
ax6.add_artist(at)


ax7 = plt.subplot(spec[7])
sm = dvel_sel_m.salem.get_map(countries=False)
sm.set_shapefile(gdir.read_shapefile('outlines'))
sm.set_shapefile(shp, color='r', linewidth=1.5)
sm.set_data(dvel_itslive.data)
sm.set_cmap('viridis')
sm.set_vmin(0)
sm.set_vmax(200)
sm.set_lonlat_contours(interval=1)
sm.visualize(ax=ax7, addcbar=False)
at = AnchoredText('h', prop=dict(size=18), frameon=True, loc='upper left')
ax7.add_artist(at)

ax8 = plt.subplot(spec[8])
sm = ds_sel.salem.get_map(countries=False)
sm.set_shapefile(gdir.read_shapefile('outlines'))
sm.set_data(smb_avg_sel)
sm.set_cmap('RdBu')
sm.set_scale_bar(location=(0.76, 0.04))
sm.set_lonlat_contours(interval=1)
sm.visualize(ax=ax8, addcbar=True)
at = AnchoredText('i', prop=dict(size=18), frameon=True, loc='upper left')
ax8.add_artist(at)

# plt.show()
plt.savefig(os.path.join(cfg.PATHS['working_dir'],
                         'data_input_plot_all.png'),
            bbox_inches='tight')
