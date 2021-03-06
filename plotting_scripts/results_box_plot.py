import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
from configobj import ConfigObj
import seaborn as sns
import pandas as pd
import numpy as np


Old_main_path = os.path.expanduser('~/k_calibration_greenland/')
MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))
old_config = ConfigObj(os.path.join(Old_main_path, 'config.ini'))


# PARAMS for plots
rcParams['axes.labelsize'] = 19
rcParams['xtick.labelsize'] = 19
rcParams['ytick.labelsize'] = 19
sns.set_context('poster')

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data/')

df_common = pd.read_csv(os.path.join(output_dir_path,
                                     'common_final_results.csv'))

# list of variables to plot
df_to_plot = df_common[[
                        'rgi_area_km2_x',
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
                        'k_racmo_upbound_calving_front_thick']]

df_to_plot.loc[:,
('k_itslive_lowbound_calving_rate')] = (df_to_plot['k_itslive_lowbound_calving_flux']*1e9)/(df_to_plot['k_itslive_lowbound_calving_front_thick']*df_to_plot['calving_front_width_x'])
df_to_plot.loc[:,
('k_itslive_value_calving_rate')] = (df_to_plot['k_itslive_value_calving_flux']*1e9)/(df_to_plot['k_itslive_value_calving_front_thick']*df_to_plot['calving_front_width_x'])
df_to_plot.loc[:,
('k_itslive_upbound_calving_rate')] = (df_to_plot['k_itslive_upbound_calving_flux']*1e9)/(df_to_plot['k_itslive_upbound_calving_front_thick']*df_to_plot['calving_front_width_x'])

df_to_plot.loc[:,
('k_measures_lowbound_calving_rate')] = (df_to_plot['k_measures_lowbound_calving_flux']*1e9)/(df_to_plot['k_measures_lowbound_calving_front_thick']*df_to_plot['calving_front_width_x'])
df_to_plot.loc[:,
('k_measures_value_calving_rate')] = (df_to_plot['k_measures_value_calving_flux']*1e9)/(df_to_plot['k_measures_value_calving_front_thick']*df_to_plot['calving_front_width_x'])
df_to_plot.loc[:,
('k_measures_upbound_calving_rate')] = (df_to_plot['k_measures_upbound_calving_flux']*1e9)/(df_to_plot['k_measures_upbound_calving_front_thick']*df_to_plot['calving_front_width_x'])

df_to_plot.loc[:,
('k_racmo_lowbound_calving_rate')] = (df_to_plot['k_racmo_lowbound_calving_flux']*1e9)/(df_to_plot['k_racmo_lowbound_calving_front_thick']*df_to_plot['calving_front_width_x'])
df_to_plot.loc[:,
('k_racmo_value_calving_rate')] = (df_to_plot['k_racmo_value_calving_flux']*1e9)/(df_to_plot['k_racmo_value_calving_front_thick']*df_to_plot['calving_front_width_x'])
df_to_plot.loc[:,
('k_racmo_upbound_calving_rate')] = (df_to_plot['k_racmo_upbound_calving_flux']*1e9)/(df_to_plot['k_racmo_upbound_calving_front_thick']*df_to_plot['calving_front_width_x'])

# Classify the glaciers by area classes
df_to_plot["area_class"] = np.digitize(df_to_plot["rgi_area_km2_x"],
                                       [0, 5, 15, 50, 1300],
                                       right=True)

# to_plot_k = df_to_plot[['calving_inversion_k_k_measures_lowbound',
#                         'calving_inversion_k_k_measures_value',
#                         'calving_inversion_k_k_measures_upbound',
#                         'calving_inversion_k_k_itslive_lowbound',
#                         'calving_inversion_k_k_itslive_value',
#                         'calving_inversion_k_k_itslive_upbound',
#                         'calving_inversion_k_k_racmo_lowbound',
#                         'calving_inversion_k_k_racmo_value',
#                         'calving_inversion_k_k_racmo_upbound',
#                         'area_class']]

# to_plot_q = df_to_plot[['calving_flux_k_measures_lowbound',
#                         'calving_flux_k_measures_value',
#                         'calving_flux_k_measures_upbound',
#                         'calving_flux_k_itslive_lowbound',
#                         'calving_flux_k_itslive_value',
#                         'calving_flux_k_itslive_upbound',
#                         'calving_flux_k_racmo_lowbound',
#                         'calving_flux_k_racmo_value',
#                         'calving_flux_k_racmo_upbound',
#                         'area_class']]
#
#
# to_plot_r = df_to_plot[['calving_rate_k_measures_lowbound',
#                         'calving_rate_k_measures_value',
#                         'calving_rate_k_measures_upbound',
#                         'calving_rate_k_itslive_lowbound',
#                         'calving_rate_k_itslive_value',
#                         'calving_rate_k_itslive_upbound',
#                         'calving_rate_k_racmo_lowbound',
#                         'calving_rate_k_racmo_value',
#                         'calving_rate_k_racmo_upbound',
#                         'area_class']]
#
to_plot_k = df_to_plot[['k_measures_value_calving_inversion_k',
                        'k_itslive_value_calving_inversion_k',
                        'k_racmo_value_calving_inversion_k',
                        'area_class']]

to_plot_q = df_to_plot[['k_measures_value_calving_flux',
                        'k_itslive_value_calving_flux',
                        'k_racmo_value_calving_flux',
                        'area_class']]


to_plot_r = df_to_plot[['k_measures_value_calving_rate',
                        'k_itslive_value_calving_rate',
                        'k_racmo_value_calving_rate',
                        'area_class']]

to_plot_q = to_plot_q.melt('area_class',
                           var_name='Method',  value_name='calving_flux')
to_plot_k = to_plot_k.melt('area_class',
                           var_name='Method',  value_name='k_value')
to_plot_r = to_plot_r.melt('area_class',
                           var_name='Method',  value_name='calving_rate')

#to_plot_k = to_plot_k.drop(to_plot_k[to_plot_k.k_value > 20].index)
to_plot_k = to_plot_k.drop(to_plot_k[to_plot_k.k_value == 0].index)

to_plot_q = to_plot_q.drop(to_plot_q[to_plot_q.calving_flux == 0].index)

to_plot_r = to_plot_r.drop(to_plot_r[to_plot_r.calving_rate == 0].index)


color_palette = sns.color_palette("deep")

# color_array = [color_palette[3], color_palette[4],
#                color_palette[2], color_palette[0],
#                color_palette[3], color_palette[4],
#                color_palette[2], color_palette[1],
color_array = [color_palette[0], color_palette[2],
               color_palette[1]]

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8))

# ax0.set_yscale("log")
g0 = sns.catplot(x="area_class", y="k_value", hue='Method',
                 data=to_plot_k, kind='box', ax=ax0, legend=True,
                 palette=color_array, showfliers=False)
ax0.set_xticklabels(labels=['0-5', '5-15', '15-50', '50-1300'])
ax0.set_xlabel('Area class [$km^2$]')
ax0.set_ylabel('$k$ [$yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=22), frameon=True, loc=2)
ax0.add_artist(at)

# replace labels
ax0.get_legend().remove()
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, ['$k_{MEaSUREs}$',
                     '$k_{ITSlive}$',
                     '$k_{RACMO}$'], loc=1, fontsize=19)



ax1.set_yscale("log")
g1 = sns.catplot(x="area_class", y="calving_flux", hue='Method',
                 data=to_plot_q, kind='box', ax=ax1, legend=True,
                 palette=color_array)
ax1.set_xticklabels(labels=['0-5', '5-15', '15-50', '50-1300'])
ax1.set_xlabel('Area class [$km^2$]')
ax1.set_ylabel('$q_{calving}$ [$km^3yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=22), frameon=True, loc=2)
ax1.add_artist(at)

# replace labels
ax1.get_legend().remove()
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, ['$q_{calving-MEaSUREs}$',
                     '$q_{calving-ITSlive}$',
                     '$q_{calving-RACMO}$'], loc=4, fontsize=19)

ax2.set_yscale("log")
g2 = sns.catplot(x="area_class", y="calving_rate", hue='Method',
                 data=to_plot_r, kind='box', ax=ax2, legend=True,
                 palette=color_array)
ax2.set_xticklabels(labels=['0-5', '5-15', '15-50', '50-1300'])
ax2.set_xlabel('Area class [$km^2$]')
ax2.set_ylabel('$r_{calving}$ [$myr^{-1}$]')
at = AnchoredText('c', prop=dict(size=22), frameon=True, loc=2)
ax2.add_artist(at)

# replace labels
ax2.get_legend().remove()
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, ['$r_{calving-MEaSUREs}$',
                     '$r_{calving-ITSlive}$',
                     '$r_{calving-RACMO}$'], loc=3, fontsize=19)

plt.close(2)
plt.close(3)
plt.close(4)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'box_plot.png'),
              bbox_inches='tight')