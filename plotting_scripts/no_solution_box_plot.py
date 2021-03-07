import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
from configobj import ConfigObj
import seaborn as sns
import pandas as pd
import numpy as np

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

from k_tools import misc

# PARAMS for plots
rcParams['axes.labelsize'] = 19
rcParams['xtick.labelsize'] = 19
rcParams['ytick.labelsize'] = 19
sns.set_context('poster')

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

# Reads racmo calibration output
output_racmo = os.path.join(MAIN_PATH,
                            config['racmo_calibration_results'])

path_no_solution = os.path.join(output_racmo,
                                      'glaciers_with_no_solution.csv')

no_sol_ids = misc.read_rgi_ids_from_csv(path_no_solution)
print(len(no_sol_ids))

# Get preprocessing data for this glaciers
all_glaciers_prepro = os.path.join(MAIN_PATH,
                                   'output_data/01_Greenland_prepo/')
df_prepro_all = pd.read_csv(os.path.join(all_glaciers_prepro,
                'glacier_statistics_greenland_no_calving_with_sliding_.csv'))

df_prepro_ice_cap = pd.read_csv(os.path.join(MAIN_PATH,
                                             config['ice_cap_prepro']))

df_prepro_all.append(df_prepro_ice_cap, ignore_index=True)

keep_indexes_no_sol = [(i in no_sol_ids) for i in df_prepro_all.rgi_id]
df_prepro_all_no_sol = df_prepro_all.iloc[keep_indexes_no_sol]

# Get velocity data for this glaciers
# Read velocity observations from MEaSURES
d_obs_measures = pd.read_csv(os.path.join(MAIN_PATH,
                                         config['processed_vel_measures']))


# Read velocity observations from ITSLive
d_obs_itslive = pd.read_csv(os.path.join(MAIN_PATH,
                                         config['processed_vel_itslive']))

no_sol_ids = df_prepro_all_no_sol.rgi_id.values

vel_M = []
error_M = []
ids_M = []

ids_no_M_data = []

for rgi_id in no_sol_ids:
    index_measures = d_obs_measures.index[d_obs_measures['RGI_ID'] == rgi_id].tolist()
    if len(index_measures) == 0:
        print('There is no Velocity data for this glacier' + rgi_id)
        ids_no_M_data = np.append(ids_no_M_data, rgi_id)
        continue
    else:
        # Perform the first step calibration and save the output as a
        # pickle file per glacier
        data_obs = d_obs_measures.iloc[index_measures]
        vel_M = np.append(vel_M, data_obs.vel_calving_front.values)
        error_M = np.append(error_M, data_obs.error_calving_front.values)
        ids_M = np.append(ids_M, data_obs.RGI_ID.values)

vel_I = []
error_I = []
ids_I = []

ids_no_I_data = []

for rgi_id in no_sol_ids:
    index_itslive = d_obs_itslive.index[d_obs_itslive['RGI_ID'] == rgi_id].tolist()
    if len(index_itslive) == 0:
        print('There is no Velocity data for this glacier' + rgi_id)
        ids_no_I_data = np.append(ids_no_I_data, rgi_id)
        continue
    else:
        # Perform the first step calibration and save the output as a
        # pickle file per glacier
        data_i = d_obs_itslive.iloc[index_itslive]
        vel_I = np.append(vel_I, data_i.vel_calving_front.values)
        error_I = np.append(error_I, data_i.error_calving_front.values)
        ids_I = np.append(ids_I, data_i.RGI_ID.values)

print(len(ids_I), len(ids_M))

keep_indexes_M = [(i in ids_M) for i in df_prepro_all_no_sol.rgi_id]
df_measures = df_prepro_all_no_sol.iloc[keep_indexes_M]

keep_indexes_I = [(i in ids_I) for i in df_prepro_all_no_sol.rgi_id]
df_itslive = df_prepro_all_no_sol.iloc[keep_indexes_I]

df_itslive.loc[:, 'velocity_calving_front_itslive'] = vel_I
df_itslive.loc[:, 'velocity_calving_front_error_itslive'] = error_I
df_itslive.loc[:, 'rgi_id_to_check_itslive'] = ids_I

df_measures.loc[:, 'velocity_calving_front_measures'] = vel_M
df_measures.loc[:, 'velocity_calving_front_error_measures'] = error_M
df_measures.loc[:, 'rgi_id_to_check_measures'] = ids_M

df_common = pd.merge(left=df_measures,
                     right=df_itslive,
                     how='inner',
                     left_on='rgi_id',
                     right_on='rgi_id')

# df_common.to_csv(os.path.join(plot_path, 'no_solution_stats.csv'))

print(len(df_common))

print(df_common.columns)

to_plot_vel = [df_common.velocity_calving_front_measures.values,
               df_common.velocity_calving_front_itslive]

print(df_common.velocity_calving_front_measures.describe())
print(df_common.velocity_calving_front_itslive.describe())

to_plot_mu = df_common.mu_star_glacierwide_x.values

to_plot_error = df_common.dem_min_elev_on_ext_x.values

color_palette = sns.color_palette("deep")

# color_array = [color_palette[3], color_palette[4],
#                color_palette[2], color_palette[0],
#                color_palette[3], color_palette[4],
#                color_palette[2], color_palette[1],
color_array = [color_palette[0], color_palette[2]]

fig, ax0 = plt.subplots(1, 1, figsize=(8, 8))

# ax0.set_yscale("log")
g0 = sns.boxplot(data=to_plot_vel, palette=color_array,
                 ax=ax0, showfliers=True)
ax0.set_xticklabels(labels=['MEaSUREs v1.0', 'ITSlive'])
ax0.set_ylabel('Surface velocity [m.$yr^{-1}$]')

plt.close(2)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'no_solution_box_plot.png'),
              bbox_inches='tight')

