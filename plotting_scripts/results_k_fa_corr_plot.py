import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
from configobj import ConfigObj
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

Old_main_path = os.path.expanduser('~/k_calibration_greenland/')
MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
sys.path.append(MAIN_PATH)

# velocity module
config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
old_config = ConfigObj(os.path.join(Old_main_path, 'config.ini'))


# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data/')

df_common = pd.read_csv(os.path.join(output_dir_path,
                                     'common_glaciers_all_methods.csv'))

area_cover_b = df_common.rgi_area_km2_x.sum()

df_common = df_common.loc[df_common.k_racmo_value_calving_inversion_k > 0]
df_common = df_common.loc[df_common.k_measures_value_calving_inversion_k < 20]

df_common['calving_front_width_x'] = df_common['calving_front_width_x']*1e-3

study_area = 28515.391
area_cover = df_common.rgi_area_km2_x.sum()

area_percentage = area_cover*100/study_area
area_percentage_b = area_cover_b*100/study_area
print('Area coverage')
print(area_percentage)
print(area_percentage_b)
print('-----------------')

# list of variables to plot
df_to_plot = df_common[['rgi_id',
                        'rgi_area_km2_x',
                        'cenlat_x',
                        'cenlon_x',
                        'calving_front_width_x',
                        'k_itslive_lowbound_calving_flux',
                        'k_itslive_lowbound_calving_inversion_k',
                        'k_itslive_lowbound_calving_front_thick',
                        'k_itslive_value_calving_flux',
                        'k_itslive_value_calving_inversion_k',
                        'k_itslive_value_calving_front_thick',
                        'k_itslive_upbound_calving_flux',
                        'k_itslive_upbound_calving_inversion_k',
                        'k_itslive_upbound_calving_front_thick',
                        'k_measures_lowbound_calving_flux',
                        'k_measures_lowbound_calving_inversion_k',
                        'k_measures_lowbound_calving_front_thick',
                        'k_measures_value_calving_flux',
                        'k_measures_value_calving_inversion_k',
                        'k_measures_value_calving_front_thick',
                        'k_measures_upbound_calving_flux',
                        'k_measures_upbound_calving_inversion_k',
                        'k_measures_upbound_calving_front_thick',
                        'k_racmo_lowbound_calving_flux',
                        'k_racmo_lowbound_calving_inversion_k',
                        'k_racmo_lowbound_calving_front_thick',
                        'k_racmo_value_calving_flux',
                        'k_racmo_value_calving_inversion_k',
                        'k_racmo_value_calving_front_thick',
                        'k_racmo_upbound_calving_flux',
                        'k_racmo_upbound_calving_inversion_k',
                        'k_racmo_upbound_calving_front_thick'
                        ]]

df_to_plot.loc[:,
('diff_k_measures_racmo')] = df_to_plot['k_measures_value_calving_inversion_k'] - df_to_plot['k_racmo_value_calving_inversion_k']
df_to_plot.loc[:,
('diff_k_itslive_racmo')] = df_to_plot['k_itslive_value_calving_inversion_k'] - df_to_plot['k_racmo_value_calving_inversion_k']


df_to_plot.loc[:,
('diff_q_measures_racmo')] = df_to_plot['k_measures_value_calving_flux'] - df_to_plot['k_racmo_value_calving_flux']
df_to_plot.loc[:,
('diff_q_itslive_racmo')] = df_to_plot['k_itslive_value_calving_flux'] - df_to_plot['k_racmo_value_calving_flux']



df_to_plot.loc[:,
('diff_k_measures_itslive')] = df_to_plot['k_measures_value_calving_inversion_k'] - df_to_plot['k_itslive_value_calving_inversion_k']
df_to_plot.loc[:,
('diff_q_measures_itslive')] = df_to_plot['k_measures_value_calving_flux'] - df_to_plot['k_itslive_value_calving_flux']


df_to_plot.loc[:,
('error_k_measures')] = df_to_plot['k_measures_upbound_calving_inversion_k'] - df_to_plot['k_measures_lowbound_calving_inversion_k']
df_to_plot.loc[:,
('error_k_itslive')] = df_to_plot['k_itslive_upbound_calving_inversion_k'] - df_to_plot['k_itslive_lowbound_calving_inversion_k']
df_to_plot.loc[:,
('error_k_racmo')] = df_to_plot['k_racmo_upbound_calving_inversion_k'] - df_to_plot['k_racmo_lowbound_calving_inversion_k']


df_to_plot.loc[:,
('error_q_measures')] = df_to_plot['k_measures_upbound_calving_flux'] - df_to_plot['k_measures_lowbound_calving_flux']
df_to_plot.loc[:,
('error_q_itslive')] =  df_to_plot['k_itslive_upbound_calving_flux'] - df_to_plot['k_itslive_lowbound_calving_flux']
df_to_plot.loc[:,
('error_q_racmo')] = df_to_plot['k_racmo_upbound_calving_flux'] - df_to_plot['k_racmo_lowbound_calving_flux']


df_measures = df_to_plot[['rgi_id',
                          'cenlat_x',
                          'cenlon_x',
                          'calving_front_width_x',
                          'k_measures_value_calving_flux',
                          'k_measures_value_calving_inversion_k',
                          'k_racmo_value_calving_flux',
                          'k_racmo_value_calving_inversion_k',
                          'diff_k_measures_racmo',
                          'diff_q_measures_racmo',
                          'error_k_measures',
                          'error_k_racmo',
                          'error_q_measures',
                          'error_q_racmo']]


df_itslive = df_to_plot[['rgi_id',
                         'cenlat_x',
                         'cenlon_x',
                         'calving_front_width_x',
                         'k_itslive_value_calving_flux',
                         'k_itslive_value_calving_inversion_k',
                         'k_racmo_value_calving_flux',
                         'k_racmo_value_calving_inversion_k',
                         'diff_k_itslive_racmo',
                         'diff_q_itslive_racmo',
                         'error_k_itslive',
                         'error_k_racmo',
                         'error_q_itslive',
                         'error_q_racmo']]

df_vel = df_to_plot[['rgi_id',
                     'cenlat_x',
                     'cenlon_x',
                     'calving_front_width_x',
                     'k_itslive_value_calving_flux',
                     'k_itslive_value_calving_inversion_k',
                     'k_measures_value_calving_flux',
                     'k_measures_value_calving_inversion_k',
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
      stats.shapiro(df_measures.k_racmo_value_calving_inversion_k.values))
print('q_calving RACMO normality test: ',
      stats.shapiro(df_measures.k_racmo_value_calving_flux.values))
print('k-value measures normality test: ',
      stats.shapiro(df_measures.k_measures_value_calving_inversion_k.values))
print('q_calving measures normality test: ',
      stats.shapiro(df_measures.k_measures_value_calving_flux.values))

print('k-value itslive normality test: ',
      stats.shapiro(df_itslive.k_itslive_value_calving_inversion_k.values))
print('q_calving itslive normality test: ',
      stats.shapiro(df_itslive.k_itslive_value_calving_flux.values))
print('----------------------')

# Pearson and kendal tests
# MEaSUREs vs RACMO
r_kendal_k_MvR, p_kendal_k_MvR = stats.kendalltau(df_measures.k_measures_value_calving_inversion_k.values,
        df_measures.k_racmo_value_calving_inversion_k.values)

r_kendal_q_MvR, p_kendal_q_MvR = stats.kendalltau(df_measures.k_measures_value_calving_flux.values,
        df_measures.k_racmo_value_calving_flux.values)

r_pearson_k_MvR, p_pearson_k_MvR = stats.pearsonr(df_measures.k_measures_value_calving_inversion_k.values,
        df_measures.k_racmo_value_calving_inversion_k.values)

r_pearson_q_MvR, p_pearson_q_MvR = stats.pearsonr(df_measures.k_measures_value_calving_flux.values,
        df_measures.k_racmo_value_calving_flux.values)

# ITSlive vs RACMO
r_kendal_k_IvR, p_kendal_k_IvR = stats.kendalltau(df_itslive.k_itslive_value_calving_inversion_k.values,
        df_itslive.k_racmo_value_calving_inversion_k.values)

r_kendal_q_IvR, p_kendal_q_IvR = stats.kendalltau(df_itslive.k_itslive_value_calving_flux.values,
        df_itslive.k_racmo_value_calving_flux.values)

r_pearson_k_IvR, p_pearson_k_IvR = stats.pearsonr(df_itslive.k_itslive_value_calving_inversion_k.values,
        df_itslive.k_racmo_value_calving_inversion_k.values)

r_pearson_q_IvR, p_pearson_q_IvR = stats.pearsonr(df_itslive.k_itslive_value_calving_flux.values,
        df_itslive.k_racmo_value_calving_flux.values)


# ITSlive vs MEaSUREs
r_kendal_k_IvM, p_kendal_k_IvM = stats.kendalltau(df_itslive.k_itslive_value_calving_inversion_k.values,
                                                  df_measures.k_measures_value_calving_inversion_k.values)

r_kendal_q_IvM, p_kendal_q_IvM = stats.kendalltau(df_itslive.k_itslive_value_calving_flux.values,
                                                  df_measures.k_measures_value_calving_flux.values)

r_pearson_k_IvM, p_pearson_k_IvM = stats.pearsonr(df_itslive.k_itslive_value_calving_inversion_k.values,
                                                  df_measures.k_measures_value_calving_inversion_k.values)

r_pearson_q_IvM, p_pearson_q_IvM = stats.pearsonr(df_itslive.k_itslive_value_calving_flux.values,
                                                  df_measures.k_measures_value_calving_flux.values)


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

df_measures.rename(columns={'k_measures_value_calving_inversion_k': 'k_value'},
                   inplace=True)

df_measures.rename(columns={'k_measures_value_calving_flux': 'q_value'},
                   inplace=True)

df_itslive.rename(columns={'k_itslive_value_calving_inversion_k': 'k_value'},
                   inplace=True)

df_itslive.rename(columns={'k_itslive_value_calving_flux': 'q_value'},
                   inplace=True)


concatenated = pd.concat([df_measures.assign(dataset='MEaSUREs vs RACMO'),
                          df_itslive.assign(dataset='ITSlive vs RACMO')])

concatenated = concatenated.drop(concatenated[concatenated.k_value == 0].index)

concatenated = concatenated.drop(concatenated[concatenated.k_racmo_value_calving_flux == 0].index)
concatenated = concatenated.drop(concatenated[concatenated.q_value == 0].index)

# Now plotting
color_palette_k = sns.color_palette("muted")
color_palette_q = sns.color_palette("deep")

# FIGURE 5
fig5 = plt.figure(figsize=(16.5, 10), constrained_layout=True)

gs = fig5.add_gridspec(2, 3, wspace=0.01, hspace=0.1)


ax0 = fig5.add_subplot(gs[0, 0])
color_array = [color_palette_k[0], color_palette_k[2]]
k_racmo = df_measures.k_racmo_value_calving_inversion_k.values
k_racmo_error = df_measures.error_k_racmo.values
k_measures = df_measures.k_value.values
k_measures_error = df_measures.error_k_measures.values
ax0.errorbar(k_racmo, k_measures,
             xerr=k_racmo_error, yerr=k_measures_error,
             color=color_palette_k[0], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax0.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax0.set_xlim(-0.2, 4)
ax0.set_ylim(-0.2, 4)
ax0.set_xticks([0, 1, 2, 3, 4])
ax0.set_xlabel('$k_{RACMO}$ [yr$^{-1}$]')
ax0.set_ylabel('$k_{MEaSUREs}$ [yr$^{-1}$]')

at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
test0 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_MvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_MvR, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1)
test0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(at)
ax0.add_artist(test0)

ax1 = fig5.add_subplot(gs[0, 1])
k_racmo = df_itslive.k_racmo_value_calving_inversion_k.values
k_racmo_error = df_itslive.error_k_racmo.values
k_itslive = df_itslive.k_value.values
k_itslive_error = df_itslive.error_k_itslive.values
ax1.errorbar(k_racmo, k_itslive,
             xerr=k_racmo_error, yerr=k_itslive_error,
             color=color_palette_k[2], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax1.plot([0, 3], [0, 3], c='grey', alpha=0.7)
ax1.set_xlim(-0.2, 4)
ax1.set_ylim(-0.2, 4)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xlabel('$k_{RACMO}$ [yr$^{-1}$]')
ax1.set_ylabel('$k_{ITSlive}$ [yr$^{-1}$]')
test1 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_IvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_IvR, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
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
ax2.set_xlim(-0.2, 4)
ax2.set_ylim(-0.2, 4)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xlabel('$k_{ITSlive}$ [yr$^{-1}$]')
ax2.set_ylabel('$k_{MEaSUREs}$ [yr$^{-1}$]')

test2 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k_IvM, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_k_IvM, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1)
test2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(test2)

ax3 = fig5.add_subplot(gs[1, 0])
q_racmo = df_measures.k_racmo_value_calving_flux.values
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
ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax3.set_xlabel('$q_{calving-RACMO}$ [$km^3$yr$^{-1}$]')
ax3.set_ylabel('$q_{calving-MEaSUREs}$ [$km^3$yr$^{-1}$]')
test3 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_MvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_MvR, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1)
test3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
ax3.add_artist(at)
ax3.add_artist(test3)


ax4 = fig5.add_subplot(gs[1, 1])
q_racmo = df_itslive.k_racmo_value_calving_flux.values
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
ax4.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax4.set_xlabel('$q_{calving-RACMO}$ [$km^3$yr$^{-1}$]')
ax4.set_ylabel('$q_{calving-ITSlive}$ [$km^3$yr$^{-1}$]')
test4 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_IvR, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_IvR, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1)
test4.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc=2)
ax4.add_artist(at)
ax4.add_artist(test4)

ax5 = fig5.add_subplot(gs[1, 2])
q_measures = df_measures.q_value.values
q_measures_error = df_measures.error_q_measures.values
q_itslive = df_itslive.q_value.values
q_itslive_error = df_itslive.error_q_itslive.values
ax5.errorbar(q_itslive, q_measures,
             xerr=q_itslive_error, yerr=q_measures_error,
             color=color_palette_q[4], fmt='o', alpha=0.5,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)

ax5.plot([0, 0.75], [0, 0.75], c='grey', alpha=0.7)
ax5.set_xlim(-0.1, 1.0)
ax5.set_ylim(-0.1, 1.0)
ax5.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax5.set_xlabel('$q_{calving-ITSlive}$ [$km^3$yr$^{-1}$]')
ax5.set_ylabel('$q_{calving-MEaSUREs}$ [$km^3$yr$^{-1}$]')
test5 = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q_IvM, ".2f")) +
                    '\np-value = ' + str(format(p_pearson_q_IvM, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1)
test5.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc=2)
ax5.add_artist(at)
ax5.add_artist(test5)


plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'k_values_fa_result_corr_no_width.pdf'),
                  bbox_inches='tight')
