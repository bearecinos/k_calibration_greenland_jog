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
from oggm import utils
import geopandas as gpd

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import misc

# velocity module
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

# Study area
study_area = 28515.391

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test/')

config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

exp_name = []
for exp in config_paths.config_path:
    name = exp.split("/")[-1]
    exp_name.append(name)

path_to_exp_files = []
for name in exp_name:
    file_path = os.path.join(output_dir_path, name + '_merge_results.csv')
    path_to_exp_files.append(file_path)


print('Getting MEaSUREs data')
print(path_to_exp_files[3])
print(path_to_exp_files[4])
print(path_to_exp_files[5])
# Gather Measures obs data vs OGGM calibrated with measures
m_low = pd.read_csv(path_to_exp_files[3])
m_up = pd.read_csv(path_to_exp_files[4])
m_value = pd.read_csv(path_to_exp_files[5])

df_to_plot_m = m_value[['rgi_id',
                        'rgi_area_km2',
                        'surface_vel_obs_measures',
                        'obs_low_bound_measures',
                        'obs_up_bound_measures',
                        'calving_flux',
                        'calving_inversion_k',
                        'velocity_surf']].copy()

df_to_plot_m = df_to_plot_m.loc[df_to_plot_m.calving_flux > 0]
df_to_plot_m = df_to_plot_m.loc[df_to_plot_m.calving_inversion_k < 20]

df_to_plot_m.loc[:,
('error_velocity_surf')] = m_up['velocity_surf'].copy() - \
                                               m_low['velocity_surf'].copy()

# Error on observations
df_to_plot_m.loc[:,
('error_measures')] = df_to_plot_m['obs_up_bound_measures'].copy() - \
                                df_to_plot_m['obs_low_bound_measures'].copy()

df_to_plot_m = df_to_plot_m.rename(columns={'error_velocity_surf': 'oggm_error_velocity_surf_k_measures'})
df_to_plot_m = df_to_plot_m.rename(columns={'velocity_surf': 'oggm_velocity_surf_k_measures'})


print('Getting ITSlive data')
print(path_to_exp_files[0])
print(path_to_exp_files[1])
print(path_to_exp_files[2])
# Gather Measures obs data vs OGGM calibrated with measures
i_low = pd.read_csv(path_to_exp_files[0])
i_up = pd.read_csv(path_to_exp_files[1])
i_value = pd.read_csv(path_to_exp_files[2])

df_to_plot_i = i_value[['rgi_id',
                        'rgi_area_km2',
                        'surface_vel_obs_itslive',
                        'obs_low_bound_itslive',
                        'obs_up_bound_itslive',
                        'calving_flux',
                        'calving_inversion_k',
                        'velocity_surf']].copy()

df_to_plot_i = df_to_plot_i.loc[df_to_plot_i.calving_flux > 0]

df_to_plot_i.loc[:,
('error_velocity_surf')] = i_up['velocity_surf'].copy() - \
                                               i_low['velocity_surf'].copy()

# Error on observations
df_to_plot_i.loc[:,
('error_itslive')] = df_to_plot_i['obs_up_bound_itslive'].copy() - \
                                 df_to_plot_i['obs_low_bound_itslive'].copy()

df_to_plot_i = df_to_plot_i.rename(columns={'error_velocity_surf': 'oggm_error_velocity_surf_k_itslive'})
df_to_plot_i = df_to_plot_i.rename(columns={'velocity_surf': 'oggm_velocity_surf_k_itslive'})


print('Getting RACMO data')
print(path_to_exp_files[6])
print(path_to_exp_files[7])
print(path_to_exp_files[8])
# Gather Measures obs data vs OGGM calibrated with measures
ra_low = pd.read_csv(path_to_exp_files[6])
ra_up = pd.read_csv(path_to_exp_files[7])
ra_value = pd.read_csv(path_to_exp_files[8])

df_to_plot_ra = ra_value[['rgi_id',
                          'rgi_area_km2',
                          'fa_racmo',
                          'racmo_low_bound',
                          'racmo_up_bound',
                          'calving_flux',
                          'velocity_surf']].copy()

df_to_plot_ra = df_to_plot_ra.loc[df_to_plot_ra.calving_flux > 0]

df_to_plot_ra.loc[:,
('error_velocity_surf')] = ra_up['velocity_surf'].copy() - \
                            ra_low['velocity_surf'].copy()

# Error on observations
df_to_plot_ra.loc[:,
('error_racmo')] = df_to_plot_ra['racmo_up_bound'].copy() - \
                   df_to_plot_ra['racmo_low_bound'].copy()

df_to_plot_ra = df_to_plot_ra.rename(columns={'error_velocity_surf': 'oggm_error_velocity_surf_k_racmo'})
df_to_plot_ra = df_to_plot_ra.rename(columns={'velocity_surf': 'oggm_velocity_surf_k_racmo'})


print('Build a common data frame for MEaSUREs results and RACMOs')
df_to_plot_MR = pd.merge(left=df_to_plot_m,
                        right=df_to_plot_ra,
                        how='inner',
                        left_on='rgi_id',
                        right_on='rgi_id')
print(df_to_plot_MR.columns)
print(len(df_to_plot_MR))


print('Build a common data frame for ITslive results and RACMOs')
df_to_plot_IR = pd.merge(left=df_to_plot_i,
                        right=df_to_plot_ra,
                        how='inner',
                        left_on='rgi_id',
                        right_on='rgi_id')
print(df_to_plot_IR.columns)
print(len(df_to_plot_IR))


# Getting statistics for MEASURES AND RACMO DATA
area_coverage_M = df_to_plot_m.rgi_area_km2.sum() / study_area * 100
z_M = np.arange(0, len(df_to_plot_m), 1)
test_M, zline_M, wline_M = misc.calculate_statistics(df_to_plot_m.surface_vel_obs_measures,
                                                     df_to_plot_m.oggm_velocity_surf_k_measures,
                                                     area_coverage_M,
                                                     z_M)
print('Stats Measures vs OGGM_k_measures', area_coverage_M)

area_coverage_MR = df_to_plot_MR.rgi_area_km2_x.sum() / study_area * 100
z_MR = np.arange(0, len(df_to_plot_m), 1)

test_MR, zline_MR, wline_MR = misc.calculate_statistics(df_to_plot_MR.surface_vel_obs_measures,
                                                     df_to_plot_MR.oggm_velocity_surf_k_racmo,
                                                     area_coverage_MR,
                                                     z_MR)
print('Stats Measures vs OGGM_k_racmo', area_coverage_MR)


# Getting statistics for ITSlive AND RACMO DATA
area_coverage_I = df_to_plot_i.rgi_area_km2.sum() / study_area * 100
z_I = np.arange(0, len(df_to_plot_m), 1)

test_I, zline_I, wline_I = misc.calculate_statistics(df_to_plot_i.surface_vel_obs_itslive,
                                                     df_to_plot_i.oggm_velocity_surf_k_itslive,
                                                     area_coverage_I,
                                                     z_I)

print('Stats ITSlvie vs OGGM_k_itslive', area_coverage_I)

area_coverage_IR = df_to_plot_IR.rgi_area_km2_x.sum() / study_area * 100
z_IR = np.arange(0, len(df_to_plot_m), 1)

test_IR, zline_IR, wline_IR = misc.calculate_statistics(df_to_plot_IR.surface_vel_obs_itslive,
                                                     df_to_plot_IR.oggm_velocity_surf_k_racmo,
                                                     area_coverage_IR,
                                                     z_IR)
print('Stats ITSlive vs OGGM_k_racmo', area_coverage_IR)

# FIGURE 5
color_palette_k = sns.color_palette("deep")

fig5 = plt.figure(figsize=(20, 5.5), constrained_layout=True)

gs = fig5.add_gridspec(1, 4, wspace=0.01)

ax0 = fig5.add_subplot(gs[0, 0])

ax0.errorbar(df_to_plot_m.surface_vel_obs_measures.values,
             df_to_plot_m.oggm_velocity_surf_k_measures.values,
             xerr=df_to_plot_m.error_measures.values,
             yerr=df_to_plot_m.oggm_error_velocity_surf_k_measures.values,
             color=color_palette_k[0], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=1.5)
ax0.set_xlim(-20, 400)
ax0.set_ylim(-20, 400)
ax0.plot(z_M, zline_M, color=color_palette_k[0])
ax0.plot(z_M, wline_M, color='grey')
ax0.set_xlabel('MEaSUREs velocities \n [m $yr^{-1}$]')
ax0.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc=2)
ax0.add_artist(at)
ax0.add_artist(test_M)


ax1 = fig5.add_subplot(gs[0, 1])
ax1.errorbar(df_to_plot_MR.surface_vel_obs_measures.values,
             df_to_plot_MR.oggm_velocity_surf_k_racmo.values,
             xerr=df_to_plot_MR.error_measures.values,
             yerr=df_to_plot_MR.oggm_error_velocity_surf_k_racmo.values,
             color=color_palette_k[1], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=2.5)
ax1.plot(z_MR, zline_MR, color=color_palette_k[1])
ax1.plot(z_MR, wline_MR, color='grey')
ax1.set_xlim(-20, 400)
ax1.set_ylim(-20, 400)
ax1.set_xlabel('MEaSUREs velocities \n [m $yr^{-1}$]')
ax1.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc=2)
ax1.add_artist(at)
ax1.add_artist(test_MR)

ax2 = fig5.add_subplot(gs[0, 2])
ax2.errorbar(df_to_plot_i.surface_vel_obs_itslive.values,
             df_to_plot_i.oggm_velocity_surf_k_itslive.values,
             xerr=df_to_plot_i.error_itslive.values,
             yerr=df_to_plot_i.oggm_error_velocity_surf_k_itslive.values,
             color=color_palette_k[2], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=2.5)
ax2.plot(z_I, zline_I, color=color_palette_k[2])
ax2.plot(z_I, wline_I, color='grey')
ax2.set_xlim(-20, 400)
ax2.set_ylim(-20, 400)
ax2.set_xlabel('ITSlive velocities \n [m $yr^{-1}$]')
ax2.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(test_I)

ax3 = fig5.add_subplot(gs[0, 3])
ax3.errorbar(df_to_plot_IR.surface_vel_obs_itslive.values,
             df_to_plot_IR.oggm_velocity_surf_k_racmo.values,
             xerr=df_to_plot_IR.error_itslive.values,
             yerr=df_to_plot_IR.oggm_error_velocity_surf_k_racmo.values,
             color=color_palette_k[1], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=2.5)
ax3.plot(z_IR, zline_IR, color=color_palette_k[1])
ax3.plot(z_IR, wline_IR, color='grey')
ax3.set_xlim(-20, 400)
ax3.set_ylim(-20, 400)
ax3.set_xlabel('ITSlive velocities \n [m $yr^{-1}$]')
ax3.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc=2)
ax3.add_artist(at)
ax3.add_artist(test_IR)
# plt.show()

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'model_analysis.png'),
                 bbox_inches='tight')
