import os
import salem
import xarray as xr
import sys
import pyproj
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from configobj import ConfigObj
from oggm import workflow, cfg, utils
from oggm.shop import its_live

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18


Old_main_path = os.path.expanduser('~/k_calibration_greenland/')
MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import utils_velocity as utils_vel

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
old_config = ConfigObj(os.path.join(Old_main_path, 'config.ini'))

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')


# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# Get glaciers that belong to the ice cap.
ice_cap = rgidf[rgidf['RGIId'].str.match('RGI60-05.10315')]
# Get the id's for filter
ice_cap_ids = ice_cap.RGIId.values

# keeping only the Ice cap
keep_indexes = [(i in ice_cap_ids) for i in rgidf.RGIId]
rgidf_ice_cap = rgidf.iloc[keep_indexes]

# The mask and geo reference data
ds_geo = xr.open_dataset(os.path.join(Old_main_path,
                                      old_config['mask_topo']),
                         decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs
# Getting hi-resolution coastline
coast_line = salem.read_shapefile(os.path.join(Old_main_path,
                                               old_config['coastline']))

ice_cap = os.path.join(MAIN_PATH, config['ice_cap_from_fabi'])

vel_file = os.path.join(Old_main_path,
                        'input_data/Velocity_golive/GRE_G0240_0000_v.tif')

# OGGM Run
cfg.initialize()
cfg.PATHS['working_dir'] = utils.get_temp_dir('ice-cap-plot')


gdirs = workflow.init_glacier_directories(['RGI60-05.10315'],
                                          from_prepro_level=3,
                                          prepro_border=10)
gdir = gdirs[0]

# Selecting a zoom portion of the topo data fitting the ice cap
ds_geo_sel = ds_geo.salem.subset(grid=gdir.grid, margin=2)

# Processing vel data
dvel = utils_vel.open_vel_raster(vel_file)

dve_sel = dvel.salem.subset(grid=gdir.grid, margin=2)

sub_mar = rgidf_ice_cap.loc[rgidf_ice_cap['TermType'] == '1']

# Plotting
fig2 = plt.figure(figsize=(18, 6), constrained_layout=False)

spec = gridspec.GridSpec(1, 3, wspace=0.5)

ax0 = plt.subplot(spec[0])
sm = ds_geo_sel.salem.get_map(countries=False)
sm.set_shapefile(gdir.read_shapefile('outlines'), color='black')
sm.set_data(ds_geo_sel.Topography)
sm.set_cmap('topo')
sm.set_scale_bar()
sm.set_lonlat_contours(interval=3)
sm.visualize(ax=ax0, cbar_title='m. above s.l.', )
at = AnchoredText('a', prop=dict(size=16), frameon=True, loc=1)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
sm = ds_geo_sel.salem.get_map(countries=False)
sm.set_shapefile(rgidf_ice_cap, color='black')
sm.set_data(ds_geo_sel.Topography)
sm.set_cmap('topo')
sm.set_scale_bar()
sm.set_lonlat_contours(interval=3)
sm.visualize(ax=ax1, cbar_title='m. above s.l.')
at = AnchoredText('b', prop=dict(size=16), frameon=True, loc=2)
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
sm = dve_sel.salem.get_map(countries=False)
sm.set_shapefile(rgidf_ice_cap, color='black')
sm.set_shapefile(sub_mar, color='r')
sm.set_data(dve_sel.data)
sm.set_cmap('viridis')
sm.set_scale_bar()
sm.set_lonlat_contours(interval=3)
sm.visualize(ax=ax2, cbar_title='m $yr^{-1}$')
at = AnchoredText('c', prop=dict(size=16), frameon=True, loc=2)
ax2.add_artist(at)
# make it nice
plt.tight_layout()
# plt.show()

plt.savefig(os.path.join(plot_path, 'ice_cap.png'), bbox_inches='tight')
