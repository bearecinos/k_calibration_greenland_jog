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
import pickle
from scipy.stats import linregress
from scipy.optimize import fsolve
import geopandas as gpd
from oggm import cfg, utils, workflow, graphics, tasks
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
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test/')

exp_dir_path = os.path.join(MAIN_PATH,
                    'output_data/05_k_exp_for_calibration/RGI60-05.00800.csv')

measures_result_path = os.path.join(MAIN_PATH,
        'output_data/06_calibration_vel_results/measures/RGI60-05.00800.pkl')

itslive_result_path = os.path.join(MAIN_PATH,
        'output_data/06_calibration_vel_results/itslive/RGI60-05.00800.pkl')

racmo_result_path = os.path.join(MAIN_PATH,
        'output_data/07_calibration_racmo_results/RGI60-05.00800.pkl')

df_common = pd.read_csv(os.path.join(output_dir_path,
                                     'common_final_results.csv'),
                        index_col='Unnamed: 0')

exp_k = pd.read_csv(exp_dir_path, index_col='Unnamed: 0')
exp_k = exp_k.drop_duplicates(subset=('calving_flux'), keep=False)

print(exp_k.columns)


df_glacier = df_common.loc[df_common.rgi_id == 'RGI60-05.00800']

with open(measures_result_path, 'rb') as handle:
    g_m = pickle.load(handle)

with open(itslive_result_path, 'rb') as handle:
    g_i = pickle.load(handle)

with open(racmo_result_path, 'rb') as handle:
    g_r = pickle.load(handle)

print('MEASURES')
print(g_m)
print('ITSLIVE')
print(g_i)
print('RACMO')
print(g_r)

# Get linear fit for OGGM Velocity data
# If there is only one model value (k, vel) we add (0,0) intercept to
# the line. Zero k and zero velocity is a valid solution
# else we take all the points found in the first calibration step
df_oggm_m = g_m['oggm_vel'][0]
df_oggm_i = g_i['oggm_vel'][0]

k_values_m = df_oggm_m.k_values.values
k_values_i = df_oggm_i.k_values.values

velocities_m = df_oggm_m.velocity_surf.values
velocities_i = df_oggm_i.velocity_surf.values

slope_M, intercept_M, r_value_M, p_value_M, std_err_M = linregress(k_values_m,
                                                                   velocities_m)

slope_I, intercept_I, r_value_I, p_value_I, std_err_I = linregress(k_values_i,
                                                                   velocities_i)

# Get linear fit for OGGM Frontal ablation data
# Get the model data from the first calibration step
df_oggm_r = g_r['oggm_racmo'][0]

k_values_r = df_oggm_r.k_values.values
calving_fluxes = df_oggm_r.calving_flux.values

# Get the equation for the model data. y = ax + b
slope_c, intercept_c, r_value_c, p_value_c, std_err_c = linregress(k_values_r,
                                                                   calving_fluxes)

# MEASURES fitting
# Observations slope, intercept. Slope here is always zero
slope_obs_m, intercept_obs_m = [0, g_m['obs_vel'][0].vel_calving_front.iloc[0]]
slope_lwl_m , intercept_lwl_m  = [0, g_m['low_lim_vel'][0][0]]
slope_upl_m , intercept_upl_m  = [0, g_m['up_lim_vel'][0][0]]

Z_value_m = misc.solve_linear_equation(slope_obs_m,
                                       intercept_obs_m,
                                       slope_M,
                                       intercept_M)

Z_lower_bound_m = misc.solve_linear_equation(slope_lwl_m,
                                             intercept_lwl_m,
                                             slope_M,
                                             intercept_M)

Z_upper_bound_m = misc.solve_linear_equation(slope_upl_m,
                                             intercept_upl_m,
                                             slope_M,
                                             intercept_M)

# ITSLIVE fitting
# Observations slope, intercept. Slope here is always zero
slope_obs_i, intercept_obs_i = [0, g_i['obs_vel'][0].vel_calving_front.iloc[0]]
slope_lwl_i , intercept_lwl_i  = [0, g_i['low_lim_vel'][0][0]]
slope_upl_i , intercept_upl_i  = [0, g_i['up_lim_vel'][0][0]]

Z_value_i = misc.solve_linear_equation(slope_obs_i,
                                       intercept_obs_i,
                                       slope_I,
                                       intercept_I)

Z_lower_bound_i = misc.solve_linear_equation(slope_lwl_i,
                                             intercept_lwl_i,
                                             slope_I,
                                             intercept_I)

Z_upper_bound_i = misc.solve_linear_equation(slope_upl_i,
                                             intercept_upl_i,
                                             slope_I,
                                             intercept_I)


# RACMO fitting
# Observations slope, intercept. Slope here is always zero
slope_obs_r, intercept_obs_r = [0,
                                g_r['obs_racmo'][0].q_calving_RACMO_mean.iloc[0]]
slope_lwl_r, intercept_lwl_r = [0, g_r['low_lim_racmo'][0][0]]
slope_upl_r, intercept_upl_r = [0, g_r['up_lim_racmo'][0][0]]

Z_value_r = misc.solve_linear_equation(slope_obs_r,
                                       intercept_obs_r,
                                       slope_c,
                                       intercept_c)

Z_lower_bound_r = misc.solve_linear_equation(slope_lwl_r,
                                             intercept_lwl_r,
                                             slope_c,
                                             intercept_c)

Z_upper_bound_r = misc.solve_linear_equation(slope_upl_r,
                                             intercept_upl_r,
                                             slope_c,
                                             intercept_c)

import matplotlib.pyplot as plt


# Now plotting
color_palette_k = sns.color_palette("muted")
color_palette_q = sns.color_palette("deep")

# FIGURE 3

fig = plt.figure(figsize=(14, 4), constrained_layout=True)

gs = fig.add_gridspec(1, 3, wspace=0.01, hspace=0.1)

ax0 = fig.add_subplot(gs[0, 0])
k_values_m_a = np.insert(k_values_m, 0, Z_lower_bound_m[0], axis=0)
k_values_m_a = np.insert(k_values_m_a, -1, Z_upper_bound_m[0], axis=0)
ax0.plot(k_values_m, velocities_m, 'o', color=color_palette_q[0], alpha=0.3,
         label='OGGM surface velocity')
ax0.plot(k_values_m_a, intercept_obs_m + slope_obs_m * k_values_m_a, '--', color='black',
         linewidth=3.0,
         label='Surface Velocity (MEaSUREs v.1.0)')
ax0.plot(k_values_m_a, intercept_lwl_m + slope_lwl_m * k_values_m_a, '-', color='grey',
         linewidth=3.0)
ax0.plot(k_values_m_a, intercept_upl_m + slope_upl_m * k_values_m_a, '-', color='grey',
         linewidth=3.0)

ax0.fill_between(k_values_m_a, Z_lower_bound_m[1], Z_upper_bound_m[1],
                 color='grey', alpha=0.3)

ax0.plot(k_values_m_a, intercept_M + slope_M * k_values_m_a, color='purple', linewidth=3.0,
         label='Fitted line', alpha=0.6)

ax0.scatter(Z_value_m[0], Z_value_m[1], marker='x', c='orange')
ax0.scatter(Z_lower_bound_m[0], Z_lower_bound_m[1], marker='x', c='red')
ax0.scatter(Z_upper_bound_m[0], Z_upper_bound_m[1], marker='x', c='brown')
ax0.set_xlabel('$k$ [yr$^{-1}$]')
ax0.set_ylabel('Velocity [m yr$^{-1}$]')
ax0.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1,
            borderaxespad=0, frameon=False, fontsize=12)
at = AnchoredText('a', prop=dict(size=15), frameon=True, loc=2)
ax0.add_artist(at)


ax1 = fig.add_subplot(gs[0, 1])
k_values_i_a = np.insert(k_values_i, 0, Z_lower_bound_i[0], axis=0)
ax1.plot(k_values_i, velocities_i, 'o', color=color_palette_q[2],
         label='OGGM surface velocity', alpha=0.6)
ax1.plot(k_values_i_a, intercept_obs_i + slope_obs_i * k_values_i_a, '--', color='black',
         linewidth=3.0,
         label='Surface Velocity (ITSlive)')
ax1.plot(k_values_i_a, intercept_lwl_i + slope_lwl_i * k_values_i_a, '-', color='grey',
         linewidth=3.0)
ax1.plot(k_values_i_a, intercept_upl_i + slope_upl_i * k_values_i_a, '-', color='grey',
         linewidth=3.0)

ax1.plot(k_values_i_a, intercept_I + slope_I * k_values_i_a, color='purple', linewidth=3.0,
         label='Fitted line', alpha=0.6)

ax1.fill_between(k_values_i_a, Z_lower_bound_i[1], Z_upper_bound_i[1],
                 color='grey', alpha=0.3)
ax1.scatter(Z_value_i[0], Z_value_i[1], marker='x', c='orange')
ax1.scatter(Z_lower_bound_i[0], Z_lower_bound_i[1], marker='x', c='red')
ax1.scatter(Z_upper_bound_i[0], Z_upper_bound_i[1], marker='x', c='brown')
ax1.set_xlabel('$k$ [yr$^{-1}$]')
ax1.set_ylabel('Velocity [m yr$^{-1}$]')

ax1.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1,
            borderaxespad=0, frameon=False, fontsize=12)

at = AnchoredText('b', prop=dict(size=15), frameon=True, loc=2)
ax1.add_artist(at)


ax2 = fig.add_subplot(gs[0, 2])
k_values_r_a = np.insert(k_values_r, -1, Z_upper_bound_r[0], axis=0)

ax2.plot(k_values_r, calving_fluxes, 'o', color=color_palette_q[1],
         label='OGGM $q_{calving}$', alpha=0.6)
ax2.plot(k_values_r_a, intercept_obs_r + slope_obs_r*k_values_r_a, '--', color='black', linewidth=3.0,
    label='RACMO derived $q_{calving}$')
ax2.plot(k_values_r_a, intercept_lwl_r + slope_lwl_r*k_values_r_a, '-', color='grey', linewidth=3.0)
ax2.plot(k_values_r_a, intercept_upl_r + slope_upl_r*k_values_r_a, '-', color='grey', linewidth=3.0)
ax2.plot(k_values_r_a, intercept_c + slope_c*k_values_r_a, color='purple', linewidth=3.0,
    label='Fitted line', alpha=0.6)

ax2.fill_between(k_values_r_a, Z_lower_bound_r[1], Z_upper_bound_r[1],
                 color='grey', alpha=0.3)

ax2.scatter(Z_value_r[0], Z_value_r[1], marker='x', c='orange')
ax2.scatter(Z_lower_bound_r[0], Z_lower_bound_r[1], marker='x', c='red')
ax2.scatter(Z_upper_bound_r[0], Z_upper_bound_r[1], marker='x', c='brown')
ax2.set_xlabel('$k$ [yr$^{-1}$]')
ax2.set_ylabel('$q_{calving}$ [$km^3yr^{-1}$]')
ax2.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1,
            borderaxespad=0, frameon=False, fontsize=12)
at = AnchoredText('c', prop=dict(size=15), frameon=True, loc=2)
ax2.add_artist(at)

# ax0.set_title("MEaSUREs")
# ax1.set_title("ITSlive")
# ax2.set_title("RACMO")
plt.tight_layout()

plt.savefig(os.path.join(plot_path, 'calibration_method.png'),
             bbox_inches='tight')