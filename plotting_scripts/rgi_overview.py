import os
import salem
import xarray as xr
import pyproj
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from configobj import ConfigObj
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import sys

Old_main_path = os.path.expanduser('~/k_calibration_greenland/')
MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
old_config = ConfigObj(os.path.join(Old_main_path, 'config.ini'))

# velocity module
from k_tools import misc

# PARAMS for plots
rcParams['axes.labelsize'] = 25
rcParams['xtick.labelsize'] = 25
rcParams['ytick.labelsize'] = 25
sns.set_context('poster')

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Reading RACMO mask
# The mask and geo reference data
ds_geo = xr.open_dataset(os.path.join(Old_main_path,
                                      old_config['mask_topo']),
                         decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs
# Getting hi-resolution coastline
coast_line = salem.read_shapefile(os.path.join(Old_main_path,
                                               old_config['coastline']))

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

rgidf = rgidf.sort_values('RGIId', ascending=True)

# Read Areas for the ice-cap computed in OGGM during
# the pre-processing runs
df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))
df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

# # Assign an area to the ice cap from OGGM to avoid errors
rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
          'Area'] = df_prepro_ic.rgi_area_km2.values

rgidf['Area'] = rgidf['Area'].astype(np.float)

rgi_area_total = rgidf['Area'].sum()

print('RGI total area, ', rgi_area_total)

rgidf.set_index('RGIId')
index = rgidf.index.values

# Get the glaciers classified by Terminus type
sub_mar = rgidf[rgidf['TermType'].isin(['1'])]
sub_lan = rgidf[rgidf['TermType'].isin(['0'])]
print('Total land terminating glaciers')
print(len(sub_lan))

# Separate the ice cap to read it later and add it to marine
# Get glaciers that belong to the ice cap.
ice_cap_land_terminating = sub_lan[sub_lan['RGIId'].str.match('RGI60-05.10315')]
ice_cap_land_area = ice_cap_land_terminating['Area'].sum()

# Get the id's for filter
ice_cap_ids = ice_cap_land_terminating.RGIId.values
print('Number of land terminating minus the ice cap')
print(len(sub_lan) - len(ice_cap_ids))
print('Number of land terminating ice cap basins')
print(len(ice_cap_ids))

# Classify Marine-terminating by connectivity
sub_no_conect = sub_mar[sub_mar['Connect'].isin([0, 1])]
sub_conect = sub_mar[sub_mar['Connect'].isin([2])]

# Make a table for the area distribution
area_per_reg = rgidf[['Area', 'TermType']].groupby('TermType').sum()

area_per_reg['Area (% of all Alaska)'] = area_per_reg['Area'] / \
                                         area_per_reg.Area.sum() * 100
area_per_reg['N Glaciers'] = rgidf.groupby('TermType').count().RGIId


area_mar = sub_mar[['Area', 'Connect']].groupby('Connect').sum()
area_mar['Area (% of all Alaska)'] = area_mar['Area'] / \
                                     area_per_reg.Area.sum() * 100

category = ['Land-terminating',
            'Tidewater strongly connected',
            'Tidewater weakly connected']

area = [area_per_reg.Area[0] - ice_cap_land_area,
        area_mar.Area[2],
        area_mar.Area[0] + area_mar.Area[1] + ice_cap_land_area]

area_percent = area / rgidf.Area.astype(np.float).sum() * 100

d = {'Category': category,
     'Area (km²)': area,
     'Area (% of Greenland)': area_percent}
ds = pd.DataFrame(data=d)

print(ds)

print('Total STUDY AREA')
study_area = sub_no_conect.Area.sum() + ice_cap_land_area
print(study_area)

# Lets combine the RGIdf's of sub no connect with ice cap land terminating
rgidf_study_area = pd.concat([sub_no_conect, ice_cap_land_terminating],
                             ignore_index=True)

# Analyse errors and data gaps
prepro_errors = pd.read_csv(os.path.join(MAIN_PATH,
                                      config['prepro_err']))
area_prepro = prepro_errors.Area.sum()


output_itslive = os.path.join(MAIN_PATH,
                              config['vel_calibration_results_itslive'])
no_itslive_data = os.path.join(output_itslive,
                               'glaciers_with_no_vel_data.csv')
no_itslive_data_df = pd.read_csv(no_itslive_data)
area_no_itslive = no_itslive_data_df.Area.sum()


output_measures = os.path.join(MAIN_PATH,
                               config['vel_calibration_results_measures'])
no_measures_data = os.path.join(output_measures,
                                'glaciers_with_no_vel_data.csv')
no_measures_data_df = pd.read_csv(no_measures_data)
area_no_measures = no_measures_data_df.Area.sum()

# Reads racmo calibration output
output_racmo = os.path.join(MAIN_PATH,
                            config['racmo_calibration_results'])
no_racmo_data = os.path.join(output_racmo,
                             'glaciers_with_no_racmo_data.csv')

no_racmo_data_df = pd.read_csv(no_racmo_data)
area_no_racmo = no_racmo_data_df.Area.sum()


no_solution = os.path.join(output_racmo, 'glaciers_with_no_solution.csv')
no_sol_df = pd.read_csv(no_solution)
area_no_solution = no_sol_df.Area.sum()

ids_with_no_data = ['RGI60-05.10878' 'RGI60-05.10997' 'RGI60-05.10998']
area_no_solution_no_data = misc.calculate_study_area(ids_with_no_data,
                                                     rgidf_study_area)

category_two = ['OGGM pre-processing errors',
                'Glaciers with no ITSLive velocity data',
                'Glaciers with no MEaSUREs data',
                'Glaciers with no RACMO DATA',
                'Glaciers with no calving solution']

areas_two = [area_prepro, area_no_itslive, area_no_measures, area_no_racmo,
             area_no_solution-area_no_solution_no_data]
area_percent_two = areas_two / study_area * 100

k = {'Category': category_two,
     'Area (km²)': areas_two,
     'Area (% of study area)': area_percent_two}
dk = pd.DataFrame(data=k)

print(dk)

print('Total study Area')
print(study_area)
print('RGI total area')
print(rgi_area_total)

world_area = 705738.793
study_area_percentage = study_area / world_area * 100
print('Percentage of world area')
print(study_area_percentage)

rgi_area_from_glims = 89717.066
print(rgi_area_from_glims-rgi_area_total)

# drgi = gpd.read_file(rgi_original_path)
# drgi_original_area = drgi['Area'].sum()
# print(drgi_original_area/world_area*100)
##############################################################################
fig = plt.figure(figsize=(14, 8), constrained_layout=False)
spec = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 1.5])

ax0 = plt.subplot(spec[0])
# Define map projections and ext.
smap = ds_geo.salem.get_map(countries=False)

# Add coastline and Ice cap outline
smap.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.7)

# Land-terminating
smap.set_shapefile(sub_lan, facecolor=sns.xkcd_rgb["grey"],
                   label='Land-terminating',
                   edgecolor=None)

# Marine-terminating
# i: Not Connected
smap.set_shapefile(rgidf_study_area, facecolor=sns.xkcd_rgb["medium blue"],
                   label='Tidewater weakly connected', linewidth=3.0,
                   edgecolor=sns.xkcd_rgb["medium blue"])

# ii: Connected
smap.set_shapefile(sub_conect, facecolor=sns.xkcd_rgb["navy blue"],
                   label='Tidewater strongly connected', linewidth=3.0,
                   edgecolor=sns.xkcd_rgb["navy blue"])

smap.set_scale_bar(location=(0.78, 0.04))
smap.visualize(ax=ax0)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])

# Plotting bar plot
N = 1
ind = np.arange(N)    # the x locations for the groups
width = 0.15       # the width of the bars: can also be len(x) sequence

Land_area = ds['Area (% of Greenland)'][0]
Marine_area = ds['Area (% of Greenland)'][1] + ds['Area (% of Greenland)'][2]
Tidewater_connected = ds['Area (% of Greenland)'][1]
Tidewater_no_connected = ds['Area (% of Greenland)'][2]

# Heights of bars1 + bars2
bars = np.add(Land_area, Tidewater_connected).tolist()


p1 = ax1.bar(ind, Land_area, width,
             color=sns.xkcd_rgb["grey"], label='Land-terminating')
p2 = ax1.bar(ind, Tidewater_connected, width, bottom=Land_area,
             color=sns.xkcd_rgb["navy blue"],
             label='Tidewater connectivity level 2')
p3 = ax1.bar(ind, Tidewater_no_connected, width, bottom=bars,
             color=sns.xkcd_rgb["medium blue"],
             label='Tidewater connectivity level 0,1 (study area)')

ax1.set_ylabel('Area (% of Greenland PGs)')
ax1.set_xticks(ind, '1')
ax1.set_xticks(ind)
ax1.set_xticklabels(['1'])

ax1.legend((p1[0], p2[0], p3[0]),
           ('Land-terminating',
            'Tidewater connectivity level 2',
            'Tidewater connectivity level 0,1 (study area)'))

ax1.get_legend().remove()

handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels,
           bbox_to_anchor=(0.55, 1.05),
           loc='upper center',
           ncol=3, fontsize=18)

at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
palette = sns.color_palette("colorblind")
# Plotting bar plot
N = 1
ind = np.arange(N)    # the x locations for the groups
width = 0.15       # the width of the bars: can also be len(x) sequence

prepro_area = dk['Area (% of study area)'][0]
no_itslive_area = dk['Area (% of study area)'][1]
no_measures_area = dk['Area (% of study area)'][2]
no_racmo_area = dk['Area (% of study area)'][3]
no_sol_area = dk['Area (% of study area)'][4]

run_area = 100 - no_sol_area - prepro_area - \
           no_itslive_area - no_measures_area - no_racmo_area

# Heights of bars1 + bars2 TODO: re do this
bars1 = np.add(run_area, no_sol_area).tolist()
bars2 = np.add(bars1, no_itslive_area).tolist()
bars3 = np.add(bars2, no_measures_area).tolist()
bars4 = np.add(bars3, no_racmo_area).tolist()
bars5 = np.add(bars4, prepro_area).tolist()

p6 = ax2.bar(ind, run_area, width, color=palette[0])

p1 = ax2.bar(ind, no_sol_area, width, bottom=run_area, color=palette[1])

p2 = ax2.bar(ind, no_itslive_area, width, bottom=bars1, color=palette[2])

p3 = ax2.bar(ind, no_measures_area, width, bottom=bars2, color=palette[3])

p4 = ax2.bar(ind, no_racmo_area, width, bottom=bars3, color=palette[4])

p5 = ax2.bar(ind, prepro_area, width, bottom=bars4, color=palette[5])

ax2.set_ylabel('Area (% of study area)')
ax2.set_xticks(ind)
ax2.set_xticklabels(['2'])

lgd = ax2.legend((p6[0], p1[0], p2[0], p3[0], p4[0], p5[0]),
                 ('Processed',
                  'Without $q_{calving}$ solution',
                  'Without ITSLive velocity',
                  'Without MEaSUREs data',
                  'Without RACMO data',
                  'OGGM errors'),
                 loc='lower right', bbox_to_anchor=(2.6, 0.0), fontsize=18)

at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(lgd)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plot_path, 'rgi_overview.png'),
            bbox_inches='tight', pad_inches=0.25)
