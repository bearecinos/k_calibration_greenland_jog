import os
import sys
from configobj import ConfigObj
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import optimize
os.getcwd()


def func(x, a, b):
    return a*np.exp(b*x)


MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# PARAMS for plots
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
sns.set_context('poster')

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test/')
climate_dir_path = os.path.join(MAIN_PATH, 'output_data/11_climate_stats/')

df_common = pd.read_csv(os.path.join(output_dir_path,
                                     'common_final_results.csv'))

df_climate = pd.read_csv(os.path.join(climate_dir_path,
                         'glaciers_PDM_temp_at_the_freeboard_height.csv'),
                         index_col='Unnamed: 0')

df_climate.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

df = pd.merge(left=df_common,
              right=df_climate,
              how='inner',
              left_on = 'rgi_id',
              right_on='rgi_id')

print(len(df))

# Select data to plot
df = df.loc[df.calving_inversion_k_k_racmo_value > 0]
df = df.loc[df.calving_inversion_k_k_measures_value < 20]

print(len(df))

# corr_matrix = df.corr(method='kendall')
# corr_matrix.to_csv(os.path.join(plot_path,
#                                  'correlation_matrix_all_kendal.csv'))

# MEaSURES
df_measures = df[['calving_mu_star_k_measures_value',
                  'calving_flux_k_measures_value',
                  'calving_front_thick_k_measures_value',
                  'calving_inversion_k_k_measures_value',
                  'calving_front_slope',
                  'calving_front_free_board',
                  'calving_front_width',
                  'total_prcp_top']].copy()

# Correlation test's
corr_measures_k_s, p_measures_k_s = stats.pearsonr(df_measures.calving_inversion_k_k_measures_value, df_measures.calving_front_slope)
corr_measures_k_f, p_measures_k_f = stats.pearsonr(df_measures.calving_inversion_k_k_measures_value, df_measures.calving_front_free_board)
corr_measures_k_m, p_measures_k_m = stats.pearsonr(df_measures.calving_inversion_k_k_measures_value, df_measures.calving_mu_star_k_measures_value)
corr_measures_k_p, p_measures_k_p = stats.pearsonr(df_measures.calving_inversion_k_k_measures_value, df_measures.total_prcp_top)

# poly fit for slope
a_M, b_M, c_M = np.polyfit(df_measures.calving_front_slope,
                           df_measures.calving_inversion_k_k_measures_value, 2)
x_M = np.linspace(0, max(df_measures.calving_front_slope), 100)

# y = ax^2 + bx + c
y_M = a_M*(x_M**2) + b_M*x_M + c_M

eq_M = '\ny = ' + str(np.around(a_M,
                                decimals=2))+'x^2 '+ str(np.around(b_M,
                                decimals=2))+'x +'+ str(np.around(c_M,
                                decimals=2))

# exponential fit for slope
initial_guess = [0.1, 1]
popt_M, pcov_M = optimize.curve_fit(func, df_measures.calving_front_slope,
                                    df_measures.calving_inversion_k_k_measures_value,
                                    initial_guess)

x_fit_M = np.linspace(0, max(df_measures.calving_front_slope), 100)

eq_exp_M = '\ny = ' + str(np.around(popt_M[0],
                                    decimals=2))+'e$^{'+ str(np.around(popt_M[1],
                                    decimals=2))+'x}$'

df_measures.rename(columns={'calving_inversion_k_k_measures_value': 'k_value'},
                   inplace=True)
df_measures.rename(columns={'calving_mu_star_k_measures_value': 'calving_mu_star'},
                   inplace=True)
df_measures['Method'] = np.repeat('MEaSUREs',
                                  len(df_measures.k_value))

# sns.set(font_scale = 0.5)
# g = sns.pairplot(df_measures)
# plt.savefig(os.path.join(plot_path, 'measures_corr_all.pdf'),
#                  bbox_inches='tight')

# ITSLIVE
df_itslive = df[['calving_mu_star_k_itslive_value',
                 'calving_flux_k_itslive_value',
                 'calving_front_thick_k_itslive_value',
                 'calving_inversion_k_k_itslive_value',
                 'calving_front_slope',
                 'calving_front_free_board',
                 'calving_front_width',
                 'total_prcp_top']].copy()

# Correlation test's
corr_itslive_k_s, p_itslive_k_s = stats.pearsonr(df_itslive.calving_inversion_k_k_itslive_value, df_itslive.calving_front_slope)
corr_itslive_k_f, p_itslive_k_f = stats.pearsonr(df_itslive.calving_inversion_k_k_itslive_value, df_itslive.calving_front_free_board)
corr_itslive_k_m, p_itslive_k_m = stats.pearsonr(df_itslive.calving_inversion_k_k_itslive_value, df_itslive.calving_mu_star_k_itslive_value)
corr_itslive_k_p, p_itslive_k_p = stats.pearsonr(df_itslive.calving_inversion_k_k_itslive_value, df_itslive.total_prcp_top)

# poly fit for slope
a_I, b_I, c_I = np.polyfit(df_itslive.calving_front_slope,
                           df_itslive.calving_inversion_k_k_itslive_value, 2)
x_I = np.linspace(0, max(df_itslive.calving_front_slope), 100)

# y = ax^2 + bx + c
y_I = a_I*(x_I**2) + b_I*x_I + c_I

eq_I = '\ny = ' + str(np.around(a_I,
                                decimals=2))+'x^2 '+ str(np.around(b_I,
                                decimals=2))+'x +'+ str(np.around(c_I,
                                decimals=2))

# exponential fit for slope
initial_guess = [0.1, 1]
popt_I, pcov_I = optimize.curve_fit(func, df_itslive.calving_front_slope,
                                    df_itslive.calving_inversion_k_k_itslive_value,
                                    initial_guess)

x_fit_I = np.linspace(0, max(df_itslive.calving_front_slope), 100)

eq_exp_I = '\ny = ' + str(np.around(popt_I[0],
                                    decimals=2))+'e$^{'+ str(np.around(popt_I[1],
                                    decimals=2))+'x}$'

df_itslive.rename(columns={'calving_inversion_k_k_itslive_value': 'k_value'},
                   inplace=True)
df_itslive.rename(columns={'calving_mu_star_k_itslive_value': 'calving_mu_star'},
                   inplace=True)
df_itslive['Method'] = np.repeat('ITSlive',
                                  len(df_itslive.k_value))

# sns.set(font_scale = 0.5)
# g = sns.pairplot(df_itslive, plot_kws=dict(color='green'))
# plt.savefig(os.path.join(plot_path, 'itslive_corr_all.pdf'),
#                  bbox_inches='tight')

# RACMO
df_racmo = df[['calving_mu_star_k_racmo_value',
                 'calving_flux_k_racmo_value',
                 'calving_front_thick_k_racmo_value',
                 'calving_inversion_k_k_racmo_value',
                 'calving_front_slope',
                 'calving_front_free_board',
                 'calving_front_width',
                 'total_prcp_top']].copy()

# Correlation test's
corr_racmo_k_s, p_racmo_k_s = stats.pearsonr(df_racmo.calving_inversion_k_k_racmo_value, df_racmo.calving_front_slope)
corr_racmo_k_f, p_racmo_k_f = stats.pearsonr(df_racmo.calving_inversion_k_k_racmo_value, df_racmo.calving_front_free_board)
corr_racmo_k_m, p_racmo_k_m = stats.pearsonr(df_racmo.calving_inversion_k_k_racmo_value, df_racmo.calving_mu_star_k_racmo_value)
corr_racmo_k_p, p_racmo_k_p = stats.pearsonr(df_racmo.calving_inversion_k_k_racmo_value, df_racmo.total_prcp_top)

# poly fit for slope
a_R, b_R, c_R = np.polyfit(df_racmo.calving_front_slope,
                           df_racmo.calving_inversion_k_k_racmo_value, 2)
x_R = np.linspace(0, max(df_racmo.calving_front_slope), 100)

# y = ax^2 + bx + c
y_R = a_R*(x_R**2) + b_R*x_R + c_R

eq_R = '\ny = ' + str(np.around(a_R,
                                decimals=2))+'x^2 '+ str(np.around(b_R,
                                decimals=2))+'x +'+ str(np.around(c_R,
                                decimals=2))

# exponential fit for slope
initial_guess = [0.1, 1]
popt_R, pcov_R = optimize.curve_fit(func, df_racmo.calving_front_slope,
                                    df_racmo.calving_inversion_k_k_racmo_value,
                                    initial_guess)

x_fit_R = np.linspace(0, max(df_racmo.calving_front_slope), 100)

eq_exp_R = '\ny = ' + str(np.around(popt_R[0],
                                    decimals=2))+'e$^{'+ str(np.around(popt_R[1],
                                    decimals=2))+'x}$'

df_racmo.rename(columns={'calving_inversion_k_k_racmo_value': 'k_value'},
                   inplace=True)
df_racmo.rename(columns={'calving_mu_star_k_racmo_value': 'calving_mu_star'},
                   inplace=True)
df_racmo['Method'] = np.repeat('RACMO',
                                  len(df_racmo.k_value))

# sns.set(font_scale = 0.5)
# g = sns.pairplot(df_racmo, plot_kws=dict(color='orange'))
# plt.savefig(os.path.join(plot_path, 'racmo_corr_all.pdf'),
#                  bbox_inches='tight')

data_all = pd.concat([df_measures, df_itslive, df_racmo], sort=False)

data_all['calving_front_width'] = data_all.loc[:,'calving_front_width']*1e-3

#Now plotting
import matplotlib.gridspec as gridspec

color_palette = sns.color_palette("muted")

# Plot Fig 1
fig1 = plt.figure(figsize=(19, 5.5), constrained_layout=True)

spec = gridspec.GridSpec(1, 4)

ax0 = plt.subplot(spec[0])
sns.scatterplot(x='calving_front_slope', y='k_value', data=data_all, hue='Method',
                ax=ax0, alpha=0.7)
# ax0.plot(x_v, y_v) # y = ax^2 + bx + c
# ax0.plot(x_r, y_r)
ax0.plot(x_fit_M, func(x_fit_M, *popt_M))
ax0.plot(x_fit_I, func(x_fit_I, *popt_I))
ax0.plot(x_fit_R, func(x_fit_R, *popt_R))
ax0.set_xlabel('calving front slope angle \n [rad]')
ax0.set_ylabel('$k$ \n [yr$^{-1}$]')
ax0.set_ylim(-0.2,4.5)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc=2)
test0 = AnchoredText(eq_exp_M +
                     eq_exp_I +
                     eq_exp_R,
                    prop=dict(size=16),
                    frameon=True, loc=9,
                    bbox_transform=ax0.transAxes)
test0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

ax0.add_artist(at)
ax0.add_artist(test0)
ax0.legend()
ax0.get_legend().remove()

ax1 = plt.subplot(spec[1])
sns.scatterplot(x='calving_front_free_board', y='k_value',
                data=data_all, hue='Method',
                ax=ax1, alpha=0.7)
ax1.set_xlabel('Free-board \n [m]')
ax1.set_ylabel('')
ax1.set_ylim(-0.2,4.5)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc=2)
test1 = AnchoredText('$r_{s}$ = '+ str(np.around(corr_measures_k_f, decimals=3)) +
                     '\np-value = ' + str(format(p_measures_k_f, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_itslive_k_f, decimals=3)) +
                     '\np-value = ' + str(format(p_itslive_k_f, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_racmo_k_f, decimals=3)) +
                     '\np-value = ' + str(format(p_racmo_k_f, ".3E")),
                     prop=dict(size=16),
                     frameon=True, loc=1,
                     bbox_transform=ax1.transAxes)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.add_artist(test1)
ax1.get_legend().remove()

ax2 = plt.subplot(spec[2])
sns.scatterplot(x='calving_mu_star', y='k_value', data=data_all, hue='Method',
                ax=ax2, alpha=0.7)
ax2.set_xlabel('$\mu^{*}$ [mm $yr^{-1}K^{-1}$]')
ax2.set_ylabel('')
ax2.set_ylim(-0.2,4.5)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc=2)
test2 = AnchoredText('$r_{s}$ = '+ str(np.around(corr_measures_k_m, decimals=3)) +
                     '\np-value = ' + str(format(p_measures_k_m, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_itslive_k_m, decimals=3)) +
                     '\np-value = ' + str(format(p_itslive_k_m, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_racmo_k_m, decimals=3)) +
                     '\np-value = ' + str(format(p_racmo_k_m, ".3E")),
                     prop=dict(size=16),
                     frameon=True, loc=1,
                     bbox_transform=ax2.transAxes)
test2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)
ax2.add_artist(test2)
ax2.get_legend().remove()

ax3 = plt.subplot(spec[3])
sns.scatterplot(x='total_prcp_top', y='k_value', data=data_all, hue='Method',
                ax=ax3, alpha=0.7)
ax3.set_xlabel('Avg. total solid prcp \n [kg m$^{-2}$ yr$^{-1}$]')
ax3.set_ylabel('')
ax3.set_ylim(-0.2,4.5)
ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc=2)
test3 = AnchoredText('$r_{s}$ = '+ str(np.around(corr_measures_k_p, decimals=3)) +
                     '\np-value = ' + str(format(p_measures_k_p, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_itslive_k_p, decimals=3)) +
                     '\np-value = ' + str(format(p_itslive_k_p, ".3E"))+
                     '\n$r_{s}$ = ' + str(np.around(corr_racmo_k_p, decimals=3)) +
                     '\np-value = ' + str(format(p_racmo_k_p, ".3E")),
                     prop=dict(size=16),
                     frameon=True, loc=1,
                     bbox_transform=ax3.transAxes)
test3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at)
ax3.add_artist(test3)
ax3.get_legend().remove()

handles, labels = ax0.get_legend_handles_labels()
fig1.legend(handles, labels, loc='center', ncol=3, fontsize=20,
            bbox_to_anchor= (0.5, 0.99),
            fancybox=False, framealpha=1, shadow=True, borderpad=1)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plot_path, 'correlation_plot_exp_fit.pdf'),
             bbox_inches='tight')