import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from configobj import ConfigObj
from collections import defaultdict
import seaborn as sns
import pandas as pd
import numpy as np
import glob
from scipy import stats
import argparse

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']
input_data_path = config['input_data_folder']
sys.path.append(MAIN_PATH)

from k_tools import misc

# Get Millan data
main_output = os.path.join(MAIN_PATH, 'output_data')
millan_path = os.path.join(main_output,
                           '03_Process_velocity_data/thickness/millan_statistics.csv')

# Get thickness for glaciers that calve! common glaciers only
marcos_data = os.path.join(MAIN_PATH, 'output_data_marco')
df_live = pd.read_csv(marcos_data+'/itslive.csv')
df_measures = pd.read_csv(marcos_data+'/measures.csv')
df_racmo = pd.read_csv(marcos_data+'/racmo.csv')

# Get every glacier of every exp!
input_data = os.path.join(MAIN_PATH, config['volume_results'])
config_paths = pd.read_csv(os.path.join(config['input_data_folder'],
                                        config['configuration_names']))

exp_name = []
all_files = []
for exp in config_paths.config_path:
    name = exp.split("/")[-1]
    exp_path = os.path.join(input_data, exp)
    exp_name.append(name)
    all_files.append(glob.glob(exp_path + "/glacier_*.csv")[0])

for f in all_files:
    print(f)

# Calculate study area
study_area = 32202.540

########## Lets put everything for volume in a single place ##########

d_volume = defaultdict(list)
for f, name in zip(all_files, exp_name):
    print(name)
    d_volume[name] = misc.combine_volume_data(f, millan_path, name)

## Add area modeled to thickness data sets
df_live = df_live.set_index('rgi_id')
df_measures = df_measures.set_index('rgi_id')
df_racmo = df_racmo.set_index('rgi_id')

to_add_itslive = d_volume['k_itslive_lowbound']
to_add_itslive = to_add_itslive.set_index('rgi_id', drop=False)[['rgi_id', 'rgi_area_km2_x', 'millan_area_km2']]

to_add_measures = d_volume['k_measures_lowbound']
to_add_measures = to_add_measures.set_index('rgi_id', drop=False)[['rgi_id', 'rgi_area_km2_x', 'millan_area_km2']]

to_add_racmo = d_volume['k_racmo_lowbound']
to_add_racmo = to_add_racmo.set_index('rgi_id', drop=False)[['rgi_id', 'rgi_area_km2_x', 'millan_area_km2']]

df_live = df_live.join(to_add_itslive)
df_measures = df_measures.join(to_add_measures)
df_racmo = df_racmo.join(to_add_racmo)

df_live_nona = df_live.dropna(subset=['H_obs'])
df_measures_nona = df_measures.dropna(subset=['H_obs'])
df_racmo_nona = df_racmo.dropna(subset=['H_obs'])

############################## Pearson tests for THICKNESS ################################################
# MEaSUREs vs OBS
r_pearson_k_MvO, p_pearson_k_MvO = stats.pearsonr(df_measures_nona.H_model_k_measures_value.values,
        df_measures_nona.H_obs.values)

# ITSlive vs OBS
r_pearson_k_IvO, p_pearson_k_IvO = stats.pearsonr(df_live_nona.H_model_k_itslive_value.values,
        df_live_nona.H_obs.values)

# RACMO vs OBS
r_pearson_k_RvO, p_pearson_k_RvO = stats.pearsonr(df_racmo_nona.H_model_k_racmo_value.values,
        df_racmo_nona.H_obs.values)

print('PEARSONS--------')
print('MEaSUREs vs OBS')
area_thick_measures = df_measures_nona['rgi_area_km2_x'].sum()*100/study_area
print('For area total of', area_thick_measures)
if p_pearson_k_MvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_MvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_MvO)

print(r_pearson_k_MvO)

print('ITSlive vs OBS')
area_thick_live = df_live_nona['rgi_area_km2_x'].sum()*100/study_area
print('For area total of', area_thick_live)
if p_pearson_k_IvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_IvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_IvO)

print(r_pearson_k_IvO)

print('RACMO vs OBS')
area_thick_racmo = df_racmo_nona['rgi_area_km2_x'].sum()*100/study_area
print('For area total of', area_thick_racmo)
if p_pearson_k_RvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_RvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_RvO)

print(r_pearson_k_RvO)

################################### Pearson tests for Volume ##############################################
# Pearson and kendal tests
# MEaSUREs vs OBS

RV_pearson_k_MvO, PV_pearson_k_MvO = stats.pearsonr(d_volume['k_measures_value'].k_measures_value_inv_volume_km3.values,
        d_volume['k_measures_value'].millan_vol_km3.values)

# ITSlive vs OBS
RV_pearson_k_IvO, PV_pearson_k_IvO = stats.pearsonr(d_volume['k_itslive_value'].k_itslive_value_inv_volume_km3.values,
        d_volume['k_itslive_value'].millan_vol_km3.values)


# RACMO vs OBS
RV_pearson_k_RvO, PV_pearson_k_RvO = stats.pearsonr(d_volume['k_racmo_value'].k_racmo_value_inv_volume_km3.values,
        d_volume['k_racmo_value'].millan_vol_km3.values)

print('PEARSONS FOR VOLUME--------')
print('MEaSUREs vs OBS')
area_vol_measures = d_volume['k_measures_lowbound']['rgi_area_km2_x'].sum()*100/study_area
print('For area total of', area_vol_measures)
area_vol_millan_me = d_volume['k_measures_lowbound']['millan_area_km2'].sum()*100/study_area
print('For area total of millan', area_vol_millan_me)

if PV_pearson_k_MvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % PV_pearson_k_MvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % PV_pearson_k_MvO)

print(RV_pearson_k_MvO)

print('ITSlive vs OBS')
area_vol_itslive = d_volume['k_itslive_lowbound']['rgi_area_km2_x'].sum()*100/study_area
print('For area total of', area_vol_itslive)
area_vol_millan_il = d_volume['k_itslive_lowbound']['millan_area_km2'].sum()*100/study_area
print('For area total of millan', area_vol_millan_il)

if PV_pearson_k_IvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % PV_pearson_k_IvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % PV_pearson_k_IvO)

print(RV_pearson_k_IvO)

print('RACMO vs OBS')
area_vol_racmo = d_volume['k_racmo_lowbound']['rgi_area_km2_x'].sum()*100/study_area
print('For area total of', area_vol_racmo)
area_vol_millan_ra = d_volume['k_racmo_lowbound']['millan_area_km2'].sum()*100/study_area
print('For area total of millan', area_vol_millan_ra)

if PV_pearson_k_RvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % PV_pearson_k_RvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % PV_pearson_k_RvO)

print(RV_pearson_k_RvO)


############# Plotting maddnes!!! ###################################################################################

color_palette_k = sns.color_palette("muted")
from mpl_toolkits.axes_grid1 import make_axes_locatable
r = 1.2

fig1 = plt.figure(figsize=(14*r, 14*r))#, constrained_layout=True)
spec = gridspec.GridSpec(3, 3, wspace=0.5, hspace=0.3,)

ax0 = plt.subplot(spec[0])
df_live_nona['model_error'] = df_live_nona['H_model_k_itslive_upbound']-\
                              df_live_nona['H_model_k_itslive_lowbound']

ax0.errorbar(df_live_nona.H_obs.values,
             df_live_nona.H_model_k_itslive_value.values,
             xerr=df_live_nona.H_error.values,
             yerr=df_live_nona['model_error'].values,
             color=color_palette_k[2], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax0.plot([0, 800], [0, 800], c='grey', alpha=0.5, linewidth=2)
ax0.set_xlim(-10, 800)
ax0.set_ylim(-10, 800)
ax0.set_xlabel('Thickness [m] \n '
               '(Millan, et al. 2022)')
ax0.set_ylabel('Thickness [m] \n '
               'OGGM-ITSLIVE')
test0 = AnchoredText('$r$ = '+ str(format(r_pearson_k_IvO, ".2f")) +
                     '\n' +
                     '$p$ = '+ str(format(p_pearson_k_IvO, ".2E")),
                    prop=dict(size=18, color=color_palette_k[2]), frameon=False, loc=1)
test0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(test0)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])
df_measures_nona['model_error'] = df_measures_nona['H_model_k_measures_upbound']-\
                                  df_measures_nona['H_model_k_measures_lowbound']

ax1.errorbar(df_measures_nona.H_obs.values,
             df_measures_nona.H_model_k_measures_value.values,
             xerr=df_measures_nona.H_error.values,
             yerr=df_measures_nona['model_error'].values,
             color=color_palette_k[0], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax1.plot([0, 800], [0, 800], c='grey', alpha=0.5, linewidth=2)
ax1.set_xlim(-10, 800)
ax1.set_ylim(-10, 800)
ax1.set_xlabel('Thickness [m] \n (Millan, et al. 2022)')
ax1.set_ylabel('Thickness [m] \n OGGM-MEaSUREs')
test1 = AnchoredText('$r$ = '+ str(format(r_pearson_k_MvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(p_pearson_k_MvO, ".2E")),
                    prop=dict(size=18, color=color_palette_k[0]), frameon=False, loc=1)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(test1)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
df_racmo_nona['model_error'] = df_racmo_nona['H_model_k_racmo_upbound']-\
                               df_racmo_nona['H_model_k_racmo_lowbound']
ax2.errorbar(df_racmo_nona.H_obs.values,
             df_racmo_nona.H_model_k_racmo_value.values,
             xerr=df_racmo_nona.H_error.values,
             yerr=df_racmo_nona['model_error'].values,
             color=color_palette_k[1], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax2.plot([0, 800], [0, 800], c='grey', alpha=0.5, linewidth=2)
ax2.set_xlim(-10, 800)
ax2.set_ylim(-10, 800)
ax2.set_xlabel('Thickness [m] \n (Millan, et al. 2022)')
ax2.set_ylabel('Thickness [m] \n OGGM-RACMO')
test2 = AnchoredText('$r$ = '+ str(format(r_pearson_k_RvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(p_pearson_k_RvO, ".2E")),
                    prop=dict(size=18, color=color_palette_k[1]), frameon=False, loc=1)
test2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(test2)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)


ax3 = plt.subplot(spec[3])
model_error = pd.DataFrame()
model_error['itslive'] = d_volume['k_itslive_upbound'].k_itslive_upbound_inv_volume_km3 -\
                         d_volume['k_itslive_lowbound'].k_itslive_lowbound_inv_volume_km3

ax3.errorbar(d_volume['k_itslive_value'].millan_vol_km3.values,
             d_volume['k_itslive_value'].k_itslive_value_inv_volume_km3.values,
             xerr=d_volume['k_itslive_value'].millan_vol_err_km3.values,
             yerr=model_error['itslive'].values, color=color_palette_k[2], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax3.plot([0, 600], [0, 600], c='grey', alpha=0.5, linewidth=2)
ax3.set_xlim(-10, 600)
ax3.set_ylim(-10, 600)
ax3.set_xlabel('Volume [km$^3$] \n (Millan, et al. 2022)')
ax3.set_ylabel('Volume [km$^3$] \n OGGM-ITSLIVE')
test3 = AnchoredText('$r$ = '+ str(format(RV_pearson_k_IvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(PV_pearson_k_IvO, ".2E")),
                    prop=dict(size=18, color=color_palette_k[2]), frameon=False, loc=1)
test3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(test3)
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc='upper left')
ax3.add_artist(at)

ax4 = plt.subplot(spec[4])

model_error = pd.DataFrame()
model_error['measures'] = d_volume['k_measures_upbound'].k_measures_upbound_inv_volume_km3 - \
                          d_volume['k_measures_lowbound'].k_measures_lowbound_inv_volume_km3

ax4.errorbar(d_volume['k_measures_value'].millan_vol_km3.values,
             d_volume['k_measures_value'].k_measures_value_inv_volume_km3.values,
             xerr=d_volume['k_measures_value'].millan_vol_err_km3.values,
             yerr=model_error['measures'].values, color=color_palette_k[0], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax4.plot([0, 600], [0, 600], c='grey', alpha=0.5, linewidth=2)
ax4.set_xlim(-10, 600)
ax4.set_ylim(-10, 600)
ax4.set_xlabel('Volume [km$^3$] \n (Millan, et al. 2022)')
ax4.set_ylabel('Volume [km$^3$] \n OGGM-MEaSUREs')
test4 = AnchoredText('$r$ = '+ str(format(RV_pearson_k_MvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(PV_pearson_k_MvO, ".2E")),
                    prop=dict(size=18, color=color_palette_k[0]), frameon=False, loc=1)
test4.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax4.add_artist(test4)
at = AnchoredText('e', prop=dict(size=18), frameon=True, loc='upper left')
ax4.add_artist(at)

ax5 = plt.subplot(spec[5])
model_error = pd.DataFrame()
model_error['racmo'] = d_volume['k_racmo_upbound'].k_racmo_upbound_inv_volume_km3 - \
                       d_volume['k_racmo_lowbound'].k_racmo_lowbound_inv_volume_km3
ax5.errorbar(d_volume['k_racmo_value'].millan_vol_km3.values,
             d_volume['k_racmo_value'].k_racmo_value_inv_volume_km3.values,
             xerr=d_volume['k_racmo_value'].millan_vol_err_km3.values,
             yerr=model_error['racmo'].values, color=color_palette_k[1], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax5.plot([0, 600], [0, 600], c='grey', alpha=0.5, linewidth=2)
ax5.set_xlim(-10, 600)
ax5.set_ylim(-10, 600)
ax5.set_xlabel('Volume [km$^3$] \n (Millan, et al. 2022)')
ax5.set_ylabel('Volume [km$^3$] \n OGGM-RACMO')
test5 = AnchoredText('$r$ = '+ str(format(RV_pearson_k_RvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(PV_pearson_k_RvO, ".2E")),
                    prop=dict(size=18, color=color_palette_k[1]), frameon=False, loc=1)
test5.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax5.add_artist(test5)
at = AnchoredText('f', prop=dict(size=18), frameon=True, loc='upper left')
ax5.add_artist(at)


# plt.savefig(os.path.join(plot_path,
#                          'correlation_thickness_end_volume.png'),
#             bbox_inches='tight', dpi=150)

plt.show()

