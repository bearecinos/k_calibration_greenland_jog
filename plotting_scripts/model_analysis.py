import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
from configobj import ConfigObj
import seaborn as sns
import pandas as pd
import numpy as np

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
sys.path.append(MAIN_PATH)

from k_tools import misc

# velocity module
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

# Calculate study area
study_area = 32202.540

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data/')

config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

exp_name = []
for exp in config_paths.config_path:
    name = exp.split("/")[-1]
    exp_name.append(name)

# Read the different data sets
df_measures = pd.read_csv(os.path.join(output_dir_path,
                                       'glaciers_measures.csv'))
z_M = np.arange(0, len(df_measures), 1)
# keep only glaciers that calve
df_measures = df_measures[df_measures.k_measures_value_calving_flux > 0]

df_itslive = pd.read_csv(os.path.join(output_dir_path,
                                      'glaciers_itslive.csv'))
z_I = np.arange(0, len(df_itslive), 1)
df_itslive = df_itslive[df_itslive.k_itslive_value_calving_flux > 0]


df_measures_racmo = pd.read_csv(os.path.join(output_dir_path,
                                        'common_glaciers_measures_racmo.csv'))

df_measures_racmo = df_measures_racmo[df_measures_racmo.k_racmo_value_calving_inversion_k > 0]
z_MR = np.arange(0, len(df_measures_racmo), 1)

df_measures_racmo  = df_measures_racmo[df_measures_racmo.k_racmo_value_calving_flux > 0]

df_itslive_racmo = pd.read_csv(os.path.join(output_dir_path,
                                         'common_glaciers_itslive_racmo.csv'))

df_itslive_racmo = df_itslive_racmo[df_itslive_racmo.k_racmo_value_calving_inversion_k > 0]
z_IR = np.arange(0, len(df_itslive_racmo), 1)

df_itslive_racmo  = df_itslive_racmo[df_itslive_racmo.k_racmo_value_calving_flux > 0]

# Getting statistics for MEASURES vs OGGM results with k calibrated with MEASURES
area_coverage_M = df_measures.rgi_area_km2.sum() / study_area * 100

#z_M = np.arange(0, len(df_measures), 1)
test_M, zline_M, wline_M = misc.calculate_statistics(df_measures.surface_vel_obs,
                                                     df_measures.k_measures_value_velocity_surf,
                                                     area_coverage_M,
                                                     z_M)
print('Stats Measures vs OGGM_k_measures', area_coverage_M)


# Getting statistics for MEASURES vs OGGM results with k calibrated with RACMO
area_coverage_MR = df_measures_racmo.rgi_area_km2_x.sum() / study_area * 100
#z_MR = np.arange(0, len(df_measures_racmo), 1)

test_MR, zline_MR, wline_MR = misc.calculate_statistics(df_measures_racmo.surface_vel_obs,
                                                        df_measures_racmo.k_racmo_value_velocity_surf,
                                                        area_coverage_MR,
                                                        z_MR)
print('Stats Measures vs OGGM_k_racmo', area_coverage_MR)


# Getting statistics for ITSLIVE vs OGGM results with k calibrated with ITSLIVE
area_coverage_I = df_itslive.rgi_area_km2.sum() / study_area * 100
#z_I = np.arange(0, len(df_itslive), 1)

test_I, zline_I, wline_I = misc.calculate_statistics(df_itslive.surface_vel_obs,
                                                     df_itslive.k_itslive_value_velocity_surf,
                                                     area_coverage_I,
                                                     z_I)

print('Stats ITSlvie vs OGGM_k_itslive', area_coverage_I)


# Getting statistics for ITSLIVE vs OGGM results with k calibrated with RACMO
area_coverage_IR = df_itslive_racmo.rgi_area_km2_x.sum() / study_area * 100
#z_IR = np.arange(0, len(df_itslive_racmo), 1)

test_IR, zline_IR, wline_IR = misc.calculate_statistics(df_itslive_racmo.surface_vel_obs,
                                                        df_itslive_racmo.k_racmo_value_velocity_surf,
                                                        area_coverage_IR,
                                                        z_IR)
print('Stats ITSlive vs OGGM_k_racmo', area_coverage_IR)


# FIGURE 5
color_palette_k = sns.color_palette("deep")

fig5 = plt.figure(figsize=(14, 12), constrained_layout=True)

gs = fig5.add_gridspec(2, 2, wspace=0.01, hspace=2.0)

ax0 = fig5.add_subplot(gs[0, 0])

ax0.errorbar(df_measures.surface_vel_obs.values,
             df_measures.k_measures_value_velocity_surf.values,
             xerr=(df_measures.obs_up_bound -
                   df_measures.obs_low_bound).values,
             yerr=(df_measures.k_measures_upbound_velocity_surf -
                   df_measures.k_measures_lowbound_velocity_surf).values,
             color=color_palette_k[0], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=1.5)
ax0.set_xlim(-20, 400)
ax0.set_ylim(-20, 400)
ax0.plot(z_M, zline_M, color=color_palette_k[0])
ax0.plot(z_M, wline_M, color='grey')
ax0.set_xlabel('MEaSUREs velocities \n [m $yr^{-1}$]')
ax0.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
ax0.set_title('OGGM calibrated with MEaSUREs vs MEaSUREs', size=16)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc=2)
ax0.add_artist(at)
ax0.add_artist(test_M)


ax1 = fig5.add_subplot(gs[0, 1])
ax1.errorbar(df_measures_racmo.surface_vel_obs.values,
             df_measures_racmo.k_racmo_value_velocity_surf.values,
             xerr=(df_measures_racmo.obs_up_bound -
                   df_measures_racmo.obs_low_bound).values,
             yerr=(df_measures_racmo.k_racmo_upbound_velocity_surf -
                   df_measures_racmo.k_racmo_lowbound_velocity_surf).values,
             color=color_palette_k[1], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=2.5)
ax1.plot(z_MR, zline_MR, color=color_palette_k[1])
ax1.plot(z_MR, wline_MR, color='grey')
ax1.set_xlim(-20, 400)
ax1.set_ylim(-20, 400)
ax1.set_xlabel('MEaSUREs velocities \n [m $yr^{-1}$]')
ax1.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
ax1.set_title('OGGM calibrated with RACMO vs MEaSUREs', size=16)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc=2)
ax1.add_artist(at)
ax1.add_artist(test_MR)

ax2 = fig5.add_subplot(gs[1, 0])
ax2.errorbar(df_itslive.surface_vel_obs.values,
             df_itslive.k_itslive_value_velocity_surf.values,
             xerr=(df_itslive.obs_up_bound -
                   df_itslive.obs_low_bound).values,
             yerr=(df_itslive.k_itslive_upbound_velocity_surf -
                   df_itslive.k_itslive_lowbound_velocity_surf).values,
             color=color_palette_k[2], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=2.5)
ax2.plot(z_I, zline_I, color=color_palette_k[2])
ax2.plot(z_I, wline_I, color='grey')
ax2.set_xlim(-20, 400)
ax2.set_ylim(-20, 400)
ax2.set_xlabel('ITSlive velocities \n [m $yr^{-1}$]')
ax2.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
ax2.set_title('OGGM calibrated with ITSlive vs ITSlive', size=16)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(test_I)

ax3 = fig5.add_subplot(gs[1, 1])
ax3.errorbar(df_itslive_racmo.surface_vel_obs.values,
             df_itslive_racmo.k_racmo_value_velocity_surf.values,
             xerr=(df_itslive_racmo.obs_up_bound -
                   df_itslive_racmo.obs_low_bound).values,
             yerr=(df_itslive_racmo.k_racmo_upbound_velocity_surf -
                   df_itslive_racmo.k_racmo_lowbound_velocity_surf).values,
             color=color_palette_k[1], fmt='o', alpha=0.7,
             ecolor=sns.xkcd_rgb["light grey"],
             elinewidth=2.5)
ax3.plot(z_IR, zline_IR, color=color_palette_k[1])
ax3.plot(z_IR, wline_IR, color='grey')
ax3.set_xlim(-20, 400)
ax3.set_ylim(-20, 400)
ax3.set_xlabel('ITSlive velocities \n [m $yr^{-1}$]')
ax3.set_ylabel('OGGM velocities \n [m $yr^{-1}$]')
ax3.set_title('OGGM calibrated with RACMO vs ITSlive', size=16)
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc=2)
ax3.add_artist(at)
ax3.add_artist(test_IR)
# plt.show()
#
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'model_analysis_without_non_calving.png'),
                 bbox_inches='tight')
