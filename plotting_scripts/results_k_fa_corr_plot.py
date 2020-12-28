import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
from configobj import ConfigObj
import seaborn as sns
import pandas as pd
import numpy as np
import salem
import xarray as xr
import pyproj
from scipy import stats
import geopandas as gpd

Old_main_path = os.path.expanduser('~/k_calibration_greenland/')
MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

# velocity module
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
old_config = ConfigObj(os.path.join(Old_main_path, 'config.ini'))


# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

# Data input for backgrounds
coast_line = salem.read_shapefile(os.path.join(Old_main_path,
                                               old_config['coastline']))
# Projection
ds_geo = xr.open_dataset(os.path.join(Old_main_path,
                                      old_config['mask_topo']),
                         decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test/')

df_common = pd.read_csv(os.path.join(output_dir_path,
                                     'common_final_results.csv'))

area_cover_b = df_common.rgi_area_km2.sum()

df_common = df_common.loc[df_common.calving_inversion_k_k_racmo_value != 0]
df_common = df_common.loc[df_common.calving_inversion_k_k_measures_value < 20]

df_common['calving_front_width'] = df_common['calving_front_width']*1e-3

study_area = 28515.391
area_cover = df_common.rgi_area_km2.sum()

area_percentage = area_cover*100/study_area
area_percentage_b = area_cover_b*100/study_area
print('Area coverage')
print(area_percentage)
print(area_percentage_b)
print('-----------------')

# list of variables to plot
df_to_plot = df_common[['rgi_id',
                        'rgi_area_km2',
                        'cenlat',
                        'cenlon',
                        'calving_front_width',
                        'calving_flux_k_itslive_lowbound',
                        'calving_front_thick_k_itslive_lowbound',
                        'calving_inversion_k_k_itslive_lowbound',
                        'calving_flux_k_itslive_value',
                        'calving_front_thick_k_itslive_value',
                        'calving_inversion_k_k_itslive_value',
                        'calving_flux_k_itslive_upbound',
                        'calving_front_thick_k_itslive_upbound',
                        'calving_inversion_k_k_itslive_upbound',
                        'calving_flux_k_measures_lowbound',
                        'calving_front_thick_k_measures_lowbound',
                        'calving_inversion_k_k_measures_lowbound',
                        'calving_flux_k_measures_value',
                        'calving_front_thick_k_measures_value',
                        'calving_inversion_k_k_measures_value',
                        'calving_flux_k_measures_upbound',
                        'calving_front_thick_k_measures_upbound',
                        'calving_inversion_k_k_measures_upbound',
                        'calving_flux_k_racmo_lowbound',
                        'calving_front_thick_k_racmo_lowbound',
                        'calving_inversion_k_k_racmo_lowbound',
                        'calving_flux_k_racmo_value',
                        'calving_front_thick_k_racmo_value',
                        'calving_inversion_k_k_racmo_value',
                        'calving_flux_k_racmo_upbound',
                        'calving_front_thick_k_racmo_upbound',
                        'calving_inversion_k_k_racmo_upbound']]

df_to_plot.loc[:,
('diff_k_measures_racmo')] = df_to_plot['calving_inversion_k_k_measures_value'] - df_to_plot['calving_inversion_k_k_racmo_value']
df_to_plot.loc[:,
('diff_k_itslive_racmo')] = df_to_plot['calving_inversion_k_k_itslive_value'] - df_to_plot['calving_inversion_k_k_racmo_value']


df_to_plot.loc[:,
('diff_q_measures_racmo')] = df_to_plot['calving_flux_k_measures_value'] - df_to_plot['calving_flux_k_racmo_value']
df_to_plot.loc[:,
('diff_q_itslive_racmo')] = df_to_plot['calving_flux_k_itslive_value'] - df_to_plot['calving_flux_k_racmo_value']



df_to_plot.loc[:,
('diff_k_measures_itslive')] = df_to_plot['calving_inversion_k_k_measures_value'] - df_to_plot['calving_inversion_k_k_itslive_value']
df_to_plot.loc[:,
('diff_q_measures_itslive')] = df_to_plot['calving_flux_k_measures_value'] - df_to_plot['calving_flux_k_itslive_value']


df_to_plot.loc[:,
('error_k_measures')] = df_to_plot['calving_inversion_k_k_measures_upbound'] - df_to_plot['calving_inversion_k_k_measures_lowbound']
df_to_plot.loc[:,
('error_k_itslive')] = df_to_plot['calving_inversion_k_k_itslive_upbound'] - df_to_plot['calving_inversion_k_k_itslive_lowbound']
df_to_plot.loc[:,
('error_k_racmo')] = df_to_plot['calving_inversion_k_k_racmo_upbound'] - df_to_plot['calving_inversion_k_k_racmo_lowbound']


df_to_plot.loc[:,
('error_q_measures')] = df_to_plot['calving_flux_k_measures_upbound'] - df_to_plot['calving_flux_k_measures_lowbound']
df_to_plot.loc[:,
('error_q_itslive')] = df_to_plot['calving_flux_k_itslive_upbound'] - df_to_plot['calving_flux_k_itslive_lowbound']
df_to_plot.loc[:,
('error_q_racmo')] = df_to_plot['calving_flux_k_racmo_upbound'] - df_to_plot['calving_flux_k_racmo_lowbound']


df_measures = df_to_plot[['rgi_id',
                          'cenlat',
                          'cenlon',
                          'calving_front_width',
                          'calving_flux_k_measures_value',
                          'calving_inversion_k_k_measures_value',
                          'calving_flux_k_racmo_value',
                          'calving_inversion_k_k_racmo_value',
                          'diff_k_measures_racmo',
                          'diff_q_measures_racmo',
                          'error_k_measures',
                          'error_k_racmo',
                          'error_q_measures',
                          'error_q_racmo']]


df_itslive = df_to_plot[['rgi_id',
                         'cenlat',
                         'cenlon',
                         'calving_front_width',
                         'calving_flux_k_itslive_value',
                         'calving_inversion_k_k_itslive_value',
                         'calving_flux_k_racmo_value',
                         'calving_inversion_k_k_racmo_value',
                         'diff_k_itslive_racmo',
                         'diff_q_itslive_racmo',
                         'error_k_itslive',
                         'error_k_racmo',
                         'error_q_itslive',
                         'error_q_racmo']]

df_vel = df_to_plot[['rgi_id',
                     'cenlat',
                     'cenlon',
                     'calving_front_width',
                     'calving_flux_k_itslive_value',
                     'calving_inversion_k_k_itslive_value',
                     'calving_flux_k_measures_value',
                     'calving_inversion_k_k_measures_value',
                     'diff_k_measures_itslive',
                     'diff_q_measures_itslive',
                     'error_k_measures',
                     'error_q_measures',
                     'error_k_itslive',
                     'error_q_itslive']]

print(len(df_measures))
print(len(df_itslive))
print(len(df_vel))

df_measures.loc[:,
('Method')] = np.repeat('MEaSUREs vs RACMO', len(df_measures))

df_itslive.loc[:,
('Method')] = np.repeat('ITSlive vs RACMO', len(df_itslive))

# Normalise test
print('PRINTING NORMALITY TEST')
print('k-value RACMO normality test: ',
      stats.shapiro(df_measures.calving_inversion_k_k_racmo_value.values))
print('q_calving RACMO normality test: ',
      stats.shapiro(df_measures.calving_flux_k_racmo_value.values))
print('k-value measures normality test: ',
      stats.shapiro(df_measures.calving_inversion_k_k_measures_value.values))
print('q_calving measures normality test: ',
      stats.shapiro(df_measures.calving_flux_k_measures_value.values))

print('k-value itslive normality test: ',
      stats.shapiro(df_itslive.calving_inversion_k_k_itslive_value.values))
print('q_calving itslive normality test: ',
      stats.shapiro(df_itslive.calving_flux_k_itslive_value.values))
print('----------------------')

# Pearson and kendal tests
# MEaSUREs vs RACMO
r_kendal_k_MvR, p_kendal_k_MvR = stats.kendalltau(df_measures.calving_inversion_k_k_measures_value.values,
        df_measures.calving_inversion_k_k_racmo_value.values)

r_kendal_q_MvR, p_kendal_q_MvR = stats.kendalltau(df_measures.calving_flux_k_measures_value.values,
        df_measures.calving_flux_k_racmo_value.values)

r_pearson_k_MvR, p_pearson_k_MvR = stats.pearsonr(df_measures.calving_inversion_k_k_measures_value.values,
        df_measures.calving_inversion_k_k_racmo_value.values)

r_pearson_q_MvR, p_pearson_q_MvR = stats.pearsonr(df_measures.calving_flux_k_measures_value.values,
        df_measures.calving_flux_k_racmo_value.values)

# ITSlive vs RACMO
r_kendal_k_IvR, p_kendal_k_IvR = stats.kendalltau(df_itslive.calving_inversion_k_k_itslive_value.values,
        df_itslive.calving_inversion_k_k_racmo_value.values)

r_kendal_q_IvR, p_kendal_q_IvR = stats.kendalltau(df_itslive.calving_flux_k_itslive_value.values,
        df_itslive.calving_flux_k_racmo_value.values)

r_pearson_k_IvR, p_pearson_k_IvR = stats.pearsonr(df_itslive.calving_inversion_k_k_itslive_value.values,
        df_itslive.calving_inversion_k_k_racmo_value.values)

r_pearson_q_IvR, p_pearson_q_IvR = stats.pearsonr(df_itslive.calving_flux_k_itslive_value.values,
        df_itslive.calving_flux_k_racmo_value.values)


# ITSlive vs MEaSUREs
r_kendal_k_IvM, p_kendal_k_IvM = stats.kendalltau(df_itslive.calving_inversion_k_k_itslive_value.values,
                                                  df_measures.calving_inversion_k_k_measures_value.values)

r_kendal_q_IvM, p_kendal_q_IvM = stats.kendalltau(df_itslive.calving_flux_k_itslive_value.values,
                                                  df_measures.calving_flux_k_measures_value.values)

r_pearson_k_IvM, p_pearson_k_IvM = stats.pearsonr(df_itslive.calving_inversion_k_k_itslive_value.values,
                                                  df_measures.calving_inversion_k_k_measures_value.values)

r_pearson_q_IvM, p_pearson_q_IvM = stats.pearsonr(df_itslive.calving_flux_k_itslive_value.values,
                                                  df_measures.calving_flux_k_measures_value.values)


print('MEaSUREs vs RACMO')
if p_pearson_k_MvR > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_MvR)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_MvR)

print(r_pearson_k_MvR)

if p_pearson_q_MvR > 0.05:
    print('q - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_q_MvR)
else:
    print('q - values are correlated (reject H0) p=%.3f' % p_pearson_q_MvR)

print(r_pearson_q_MvR)
print('----------------------')


print('ITSlive vs RACMO')
if p_pearson_k_IvR > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_IvR)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_IvR)

print(r_pearson_k_IvR)

if p_pearson_q_IvR > 0.05:
    print('q - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_q_IvR)
else:
    print('q - values are correlated (reject H0) p=%.3f' % p_pearson_q_IvR)

print(r_pearson_q_IvR)

print('----------------------')
print('ITSlive vs MEaSUREs')
if p_pearson_k_IvM > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_IvM)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_IvM)

print(r_pearson_k_IvM)

if p_pearson_q_IvM > 0.05:
    print('q - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_q_IvM)
else:
    print('q - values are correlated (reject H0) p=%.3f' % p_pearson_q_IvM)

print(r_pearson_q_IvM)
print('----------------------')

df_measures.rename(columns={'calving_inversion_k_k_measures_value': 'k_value'},
                   inplace=True)

df_measures.rename(columns={'calving_flux_k_measures_value': 'q_value'},
                   inplace=True)

df_itslive.rename(columns={'calving_inversion_k_k_itslive_value': 'k_value'},
                   inplace=True)

df_itslive.rename(columns={'calving_flux_k_itslive_value': 'q_value'},
                   inplace=True)


concatenated = pd.concat([df_measures.assign(dataset='MEaSUREs vs RACMO'),
                          df_itslive.assign(dataset='ITSlive vs RACMO')])

concatenated = concatenated.drop(concatenated[concatenated.k_value == 0].index)

concatenated = concatenated.drop(concatenated[concatenated.calving_flux_k_racmo_value == 0].index)
concatenated = concatenated.drop(concatenated[concatenated.q_value == 0].index)

# Now plotting
color_palette_k = sns.color_palette("muted")
color_palette_q = sns.color_palette("deep")

# FIGURE 3

fig3 = plt.figure(figsize=(19, 5.5), constrained_layout=True)

gs = fig3.add_gridspec(1, 4, wspace=0.01, hspace=0.1)


ax0 = fig3.add_subplot(gs[0, 0])
color_array = [color_palette_k[0], color_palette_k[2]]
g0 = sns.scatterplot(x='calving_inversion_k_k_racmo_value',
                     y='k_value',
                     data=concatenated,
                     hue='Method',
                     palette=color_array,
                     alpha=0.8, size='calving_front_width',
                     sizes=(100, 1000), ax=ax0,
                     legend='brief')
ax0.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax0.set_xlabel('$k_{RACMO}$ [yr$^{-1}$]')
ax0.set_ylabel('$k_{velocity}$ [yr$^{-1}$]')

handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles=handles[3:-1], labels=['width [km]', '0.01', '10', '20'],
           scatterpoints=1, labelspacing=1.3,
           frameon=False, loc=4, fontsize=15,
           title_fontsize=15)

label_two = ['MEaSUREs vs RACMO','ITSlive vs RACMO']

at = AnchoredText('a', prop=dict(size=15), frameon=True, loc=2)
test0 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_MvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_MvR, ".3E")) +
                    '\n$r_{s}$ = '+ str(format(r_pearson_k_IvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_IvR, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at)
ax0.add_artist(test0)

ax1 = fig3.add_subplot(gs[0, 1])
color_array = [color_palette_q[0], color_palette_q[2]]
g1 = sns.scatterplot(x='calving_flux_k_racmo_value',
                     y='q_value',
                     data=concatenated,
                     hue='Method',
                     palette=color_array,
                     alpha=0.8, size='calving_front_width',
                     sizes=(100, 1000), ax=ax1,
                     legend='brief')
ax1.plot([0, 1.0], [0, 1.0], c='grey', alpha=0.7)
ax1.set_xlabel('$q_{calving-RACMO}$ [$km^3$yr$^{-1}$]')
ax1.set_ylabel('$q_{calving-velocity}$ [$km^3$yr$^{-1}$]')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles[3:-1], labels=['width [km]', '0.01', '10', '20'],
           scatterpoints=1, labelspacing=1.3,
           frameon=False, loc=4, fontsize=15,
           title_fontsize=15)

at = AnchoredText('b', prop=dict(size=15), frameon=True, loc=2)
test1 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_MvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_MvR, ".3E")) +
                    '\n$r_{s}$ = '+ str(format(r_pearson_q_IvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_IvR, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.add_artist(test1)


ax2 = fig3.add_subplot(gs[0, 2])
g2 = sns.scatterplot(x='calving_inversion_k_k_itslive_value',
                     y='calving_inversion_k_k_measures_value',
                     data=df_vel, color=color_palette_k[4],
                     alpha=0.8, size='calving_front_width',
                     sizes=(100, 1000), ax=ax2,
                     legend='brief')
ax2.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax2.set_xlabel('$k_{ITSlive}$ [yr$^{-1}$]')
ax2.set_ylabel('$k_{MEaSUREs}$ [yr$^{-1}$]')

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[0:-1], labels=['width [km]', '0.01', '10', '20'],
           scatterpoints=1, labelspacing=1.3,
           frameon=False, loc=4, fontsize=15,
           title_fontsize=15)
at = AnchoredText('c', prop=dict(size=15), frameon=True, loc=2)
test2 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_IvM, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_IvM, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(test2)
ax2.add_artist(at)

ax3 = fig3.add_subplot(gs[0, 3])
g3 = sns.scatterplot(x='calving_flux_k_itslive_value',
                     y='calving_flux_k_measures_value',
                     data=df_vel, color=color_palette_k[5],
                     alpha=0.8, size='calving_front_width',
                     sizes=(100, 1000), ax=ax3,
                     legend='brief')
ax3.plot([0, 1.0], [0, 1.0], c='grey', alpha=0.7)
ax3.set_xlabel('$q_{calving-ITSlive}$ [$km^3$yr$^{-1}$]')
ax3.set_ylabel('$q_{calving-MEaSUREs}$ [$km^3$yr$^{-1}$]')

handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles=handles[0:-1], labels=['width [km]', '0.01', '10', '20'],
           scatterpoints=1, labelspacing=1.3,
           frameon=False, loc=4, fontsize=15,
           title_fontsize=15)

test3 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_IvM, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_IvM, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(test3)
at = AnchoredText('d', prop=dict(size=15), frameon=True, loc=2)
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'k_values_fa_result.pdf'),
                 bbox_inches='tight')
# plt.show()

# FIGURE 4

fig4 = plt.figure(figsize=(12, 16), constrained_layout=True)

gs = fig4.add_gridspec(5, 3, wspace=0.01, hspace=0.005)

ax0 = fig4.add_subplot(gs[0:2, 0])
# Get data to plot
lon = df_measures.cenlon.values
lat = df_measures.cenlat.values
diff_k_MR = np.abs(df_measures.diff_k_measures_racmo.values)
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)
ax0.scatter(xx, yy, 200*diff_k_MR, alpha=0.5, color=color_palette_k[0],
                                        edgecolor=color_palette_k[0])
# make legend with dummy points
for a in [0.1, 0.5, 1.0]:
    ax0.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=200*a,
                label=str(a))
ax0.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$k$ differences \n [yr$^{-1}$]',
           title_fontsize=14);
sm.set_scale_bar(location=(0.8, 0.02))
sm.visualize(ax=ax0)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)


ax1 = fig4.add_subplot(gs[0:2, 1])
lon = df_itslive.cenlon.values
lat = df_itslive.cenlat.values
diff_k_IR = np.abs(df_itslive.diff_k_itslive_racmo.values)
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)
ax1.scatter(xx, yy, 200*diff_k_IR, alpha=0.5, color=color_palette_k[2],
                                        edgecolor=color_palette_k[2])
# make legend with dummy points
for a in [0.1, 0.5, 1.0]:
    ax1.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=200*a,
                label=str(a))
ax1.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$k$ differences \n [yr$^{-1}$]',
           title_fontsize=14);
sm.visualize(ax=ax1)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
ax1.add_artist(at)


ax2 = fig4.add_subplot(gs[0:2, 2])
lon = df_vel.cenlon.values
lat = df_vel.cenlat.values
diff_k_MI = np.abs(df_vel.diff_k_measures_itslive.values)
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)
ax2.scatter(xx, yy, 200*diff_k_MI, alpha=0.5, color=color_palette_k[4],
                                        edgecolor=color_palette_k[4])
# make legend with dummy points
for a in [0.1, 0.5, 1.0]:
    ax2.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=200*a,
                label=str(a))
ax2.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$k$ differences \n [yr$^{-1}$]',
           title_fontsize=14);
sm.visualize(ax=ax2)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)

ax3 = fig4.add_subplot(gs[2:4, 0])
lon = df_measures.cenlon.values
lat = df_measures.cenlat.values
diff_q_MR = np.abs(df_measures.diff_q_measures_racmo.values)

sm = ds_geo.salem.get_map(countries=False)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)
ax3.scatter(xx, yy, 1000*diff_q_MR, alpha=0.5, color=color_palette_q[0],
                                        edgecolor=color_palette_q[0])

for a in [0.05, 0.1, 0.5]:
    ax3.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=1000*a,
                label=str(a))
ax3.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$q_{calving}$ \n differences \n [km$^{3}$yr$^{-1}$]',
           title_fontsize=14);
sm.visualize(ax=ax3)
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
ax3.add_artist(at)


ax4 = fig4.add_subplot(gs[2:4, 1])
lon = df_itslive.cenlon.values
lat = df_itslive.cenlat.values
diff_q_IR = np.abs(df_itslive.diff_q_itslive_racmo.values)

sm = ds_geo.salem.get_map(countries=False)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)
ax4.scatter(xx, yy, 1000*diff_q_IR, alpha=0.5, color=color_palette_q[2],
                                        edgecolor=color_palette_q[2])

for a in [0.05, 0.1, 0.5]:
    ax4.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=1000*a,
                label=str(a))
ax4.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$q_{calving}$ \n differences \n [km$^{3}$yr$^{-1}$]',
           title_fontsize=14);
sm.visualize(ax=ax4)
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc=2)
ax4.add_artist(at)


ax5 = fig4.add_subplot(gs[2:4, 2])
lon = df_vel.cenlon.values
lat = df_vel.cenlat.values
diff_q_MI = np.abs(df_vel.diff_q_measures_itslive.values)
sm = ds_geo.salem.get_map(countries=False)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)
ax5.scatter(xx, yy, 1000*diff_q_MI, alpha=0.5, color=color_palette_k[5],
                                        edgecolor=color_palette_k[5])

for a in [0.05, 0.1, 0.5]:
    ax5.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=1000*a,
                label=str(a))
ax5.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$q_{calving}$ \n differences \n [km$^{3}$yr$^{-1}$]',
           title_fontsize=14);
sm.visualize(ax=ax5)
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc=2)
ax5.add_artist(at)

plt.tight_layout()
#
plt.savefig(os.path.join(plot_path, 'k_fa_differences.pdf'),
                  bbox_inches='tight')
# plt.show()

# FIGURE 5
fig5 = plt.figure(figsize=(16, 10), constrained_layout=True)

gs = fig5.add_gridspec(2, 3, wspace=0.01, hspace=0.1)


ax0 = fig5.add_subplot(gs[0, 0])
color_array = [color_palette_k[0], color_palette_k[2]]
k_racmo = df_measures.calving_inversion_k_k_racmo_value.values
k_racmo_error = df_measures.error_k_racmo.values
k_measures = df_measures.k_value.values
k_measures_error = df_measures.error_k_measures.values
ax0.errorbar(k_racmo, k_measures,
             xerr=k_racmo_error, yerr=k_measures_error,
             color=color_palette_k[0], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax0.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax0.set_xlim(-0.1, 4)
ax0.set_ylim(-0.1, 4)
ax0.set_xlabel('$k_{RACMO}$ [yr$^{-1}$]')
ax0.set_ylabel('$k_{MEaSUREs}$ [yr$^{-1}$]')

at = AnchoredText('a', prop=dict(size=15), frameon=True, loc=2)
test0 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_MvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_MvR, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at)
ax0.add_artist(test0)

ax1 = fig5.add_subplot(gs[0, 1])
k_racmo = df_itslive.calving_inversion_k_k_racmo_value.values
k_racmo_error = df_itslive.error_k_racmo.values
k_itslive = df_itslive.k_value.values
k_itslive_error = df_itslive.error_k_itslive.values
ax1.errorbar(k_racmo, k_itslive,
             xerr=k_racmo_error, yerr=k_itslive_error,
             color=color_palette_k[2], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax1.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax1.set_xlim(-0.1, 4)
ax1.set_ylim(-0.1, 4)
ax1.set_xlabel('$k_{RACMO}$ [yr$^{-1}$]')
ax1.set_ylabel('$k_{ITSlive}$ [yr$^{-1}$]')
test1 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_IvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_IvR, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('b', prop=dict(size=15), frameon=True, loc=2)
ax1.add_artist(at)
ax1.add_artist(test1)

ax2 = fig5.add_subplot(gs[0, 2])
k_measures = df_measures.k_value.values
k_measures_error = df_measures.error_k_measures.values
k_itslive = df_itslive.k_value.values
k_itslive_error = df_itslive.error_k_itslive.values

ax2.errorbar(k_itslive, k_measures,
             xerr=k_itslive_error, yerr=k_measures_error,
             color=color_palette_k[4], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax2.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax2.set_xlim(-0.1, 4)
ax2.set_ylim(-0.1, 4)
ax2.set_xlabel('$k_{ITSlive}$ [yr$^{-1}$]')
ax2.set_ylabel('$k_{MEaSUREs}$ [yr$^{-1}$]')

test2 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_IvM, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_IvM, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('c', prop=dict(size=15), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(test2)

ax3 = fig5.add_subplot(gs[1, 0])
q_racmo = df_measures.calving_flux_k_racmo_value.values
q_racmo_error = df_measures.error_q_racmo.values
q_measures = df_measures.q_value.values
q_measures_error = df_measures.error_q_measures.values
ax3.errorbar(q_racmo, q_measures,
             xerr=q_racmo_error, yerr=q_measures_error,
             color=color_palette_q[0], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)

ax3.plot([0, 0.75], [0, 0.75], c='grey', alpha=0.7)
ax3.set_xlim(-0.1, 1.0)
ax3.set_ylim(-0.1, 1.0)
ax3.set_xlabel('$q_{calving-RACMO}$ [$km^3$yr$^{-1}$]')
ax3.set_ylabel('$q_{calving-MEaSUREs}$ [$km^3$yr$^{-1}$]')
test3 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_MvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_MvR, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('d', prop=dict(size=15), frameon=True, loc=2)
ax3.add_artist(at)
ax3.add_artist(test3)


ax4 = fig5.add_subplot(gs[1, 1])
q_racmo = df_itslive.calving_flux_k_racmo_value.values
q_racmo_error = df_itslive.error_q_racmo.values
q_itslive = df_itslive.q_value.values
q_itslive_error = df_itslive.error_q_itslive.values
ax4.errorbar(q_racmo, q_itslive,
             xerr=q_racmo_error, yerr=q_itslive_error,
             color=color_palette_q[2], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)

ax4.plot([0, 0.75], [0, 0.75], c='grey', alpha=0.7)
ax4.set_xlim(-0.1, 1.0)
ax4.set_ylim(-0.1, 1.0)
ax4.set_xlabel('$q_{calving-RACMO}$ [$km^3$yr$^{-1}$]')
ax4.set_ylabel('$q_{calving-ITSlive}$ [$km^3$yr$^{-1}$]')
test4 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_IvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_IvR, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test4.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('e', prop=dict(size=15), frameon=True, loc=2)
ax4.add_artist(at)
ax4.add_artist(test4)

ax5 = fig5.add_subplot(gs[1, 2])
q_measures = df_measures.q_value.values
q_measures_error = df_measures.error_q_measures.values
q_itslive = df_itslive.q_value.values
q_itslive_error = df_itslive.error_q_itslive.values
ax5.errorbar(q_itslive, q_measures,
             xerr=q_itslive_error, yerr=q_measures_error,
             color=color_palette_k[5], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)

ax5.plot([0, 0.75], [0, 0.75], c='grey', alpha=0.7)
ax5.set_xlim(-0.1, 1.0)
ax5.set_ylim(-0.1, 1.0)
ax5.set_xlabel('$q_{calving-ITSlive}$ [$km^3$yr$^{-1}$]')
ax5.set_ylabel('$q_{calving-MEaSUREs}$ [$km^3$yr$^{-1}$]')
test5 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_IvM, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_IvM, ".3E")),
                    prop=dict(size=15), frameon=True, loc=1)
test5.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('f', prop=dict(size=15), frameon=True, loc=2)
ax5.add_artist(at)
ax5.add_artist(test5)


plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'k_values_fa_result_corr_no_width.png'),
                 bbox_inches='tight')
# plt.show()