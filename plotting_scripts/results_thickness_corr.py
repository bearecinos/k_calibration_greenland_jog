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

marcos_data = os.path.join(MAIN_PATH, 'output_data_marco')

df_live = pd.read_csv(marcos_data+'/itslive.csv')
df_measures = pd.read_csv(marcos_data+'/measures.csv')
df_racmo = pd.read_csv(marcos_data+'/racmo.csv')

df_live = df_live.loc[df_live.H_obs > 0]
df_measures = df_measures.loc[df_measures.H_obs > 0]
df_racmo = df_racmo.loc[df_racmo.H_obs > 0]

df_live.loc[:,
('model_error_live')] = df_live['H_model_k_itslive_upbound'] - df_live['H_model_k_itslive_lowbound']

df_measures.loc[:,
('model_error_measures')] = df_measures['H_model_k_measures_upbound'] - df_measures['H_model_k_measures_lowbound']

df_racmo.loc[:,
('model_error_racmo')] = df_racmo['H_model_k_racmo_upbound'] - df_racmo['H_model_k_racmo_lowbound']

# Pearson and kendal tests
# MEaSUREs vs OBS
r_kendal_k_MvO, p_kendal_k_MvO = stats.kendalltau(df_measures.H_model_k_measures_value.values,
        df_measures.H_obs.values)

r_pearson_k_MvO, p_pearson_k_MvO = stats.pearsonr(df_measures.H_model_k_measures_value.values,
        df_measures.H_obs.values)

# ITSlive vs OBS
r_kendal_k_IvO, p_kendal_k_IvO = stats.kendalltau(df_live.H_model_k_itslive_value.values,
        df_live.H_obs.values)

r_pearson_k_IvO, p_pearson_k_IvO = stats.pearsonr(df_live.H_model_k_itslive_value.values,
        df_live.H_obs.values)


# RACMO vs OBS
r_kendal_k_RvO, p_kendal_k_RvO = stats.kendalltau(df_racmo.H_model_k_racmo_value.values,
        df_racmo.H_obs.values)

r_pearson_k_RvO, p_pearson_k_RvO = stats.pearsonr(df_racmo.H_model_k_racmo_value.values,
        df_racmo.H_obs.values)


print('PEARSONS--------')
print('MEaSUREs vs OBS')
if p_pearson_k_MvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_MvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_MvO)

print(r_pearson_k_MvO)

print('ITSlive vs OBS')
if p_pearson_k_IvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_IvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_IvO)

print(r_pearson_k_IvO)

print('RACMO vs OBS')
if p_pearson_k_RvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k_RvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k_RvO)

print(r_pearson_k_RvO)


print('KENDALS--------')
print('MEaSUREs vs OBS')
if p_kendal_k_MvO > 0.05:
    print('q - values are uncorrelated (fail to reject H0) p=%.4f' % p_kendal_k_MvO)
else:
    print('q - values are correlated (reject H0) p=%.3f' % p_kendal_k_MvO)

print(r_kendal_k_MvO)

print('ITSlive vs OBS')
if p_kendal_k_IvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_kendal_k_IvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_kendal_k_IvO)
print(r_kendal_k_IvO)

print('RACMO vs OBS')
if p_kendal_k_RvO > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_kendal_k_RvO)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_kendal_k_RvO)
print(r_kendal_k_RvO)

print('----------------------')

# PARAMS for plots
sns.set_context('poster')

# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

color_palette_k = sns.color_palette("muted")
color_palette_q = sns.color_palette("deep")
fig5 = plt.figure(figsize=(14, 8), constrained_layout=True)

gs = fig5.add_gridspec(2, 3, wspace=0.01, hspace=0.1)


ax0 = fig5.add_subplot(gs[0, 0])
ax0.errorbar(df_live.H_obs.values, df_live.H_model_k_itslive_value.values, xerr=df_live.H_error.values,
             yerr=df_live.model_error_live.values, color=color_palette_k[2], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax0.plot([0, 800], [0, 800], c='grey', alpha=0.1)
ax0.set_xlabel('Thickness [m] \n (Millan, et al. 2022)')
ax0.set_ylabel('Thickness [m] \n OGGM-ITSLIVE')
test0 = AnchoredText('$r$ = '+ str(format(r_pearson_k_IvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(p_pearson_k_IvO, ".2E")),
                    prop=dict(size=12, color=color_palette_k[2]), frameon=False, loc=1)
test0.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax0.add_artist(test0)
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = fig5.add_subplot(gs[0, 1])
ax1.errorbar(df_measures.H_obs.values, df_measures.H_model_k_measures_value.values, xerr=df_measures.H_error.values,
             yerr=df_measures.model_error_measures.values, color=color_palette_k[0], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax1.plot([0, 800], [0, 800], c='grey', alpha=0.1)
ax1.set_xlabel('Thickness [m] \n (Millan, et al. 2022)')
ax1.set_ylabel('Thickness [m] \n OGGM-MEaSUREs')
test1 = AnchoredText('$r$ = '+ str(format(r_pearson_k_MvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(p_pearson_k_MvO, ".2E")),
                    prop=dict(size=12, color=color_palette_k[0]), frameon=False, loc=1)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(test1)
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc=2)
ax1.add_artist(at)


ax2 = fig5.add_subplot(gs[0, 2])
ax2.errorbar(df_racmo.H_obs.values, df_racmo.H_model_k_racmo_value.values, xerr=df_racmo.H_error.values,
             yerr=df_racmo.model_error_racmo.values, color=color_palette_k[3], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=2.5)
ax2.plot([0, 800], [0, 800], c='grey', alpha=0.1)
ax2.set_xlabel('Thickness [m] \n (Millan, et al. 2022)')
ax2.set_ylabel('Thickness [m] \n OGGM-RACMO')
test2 = AnchoredText('$r$ = '+ str(format(r_pearson_k_RvO, ".2f")) + '\n' +
                     '$p$ = '+ str(format(p_pearson_k_RvO, ".2E")),
                    prop=dict(size=12, color=color_palette_k[3]), frameon=False, loc=1)
test2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(test2)
at = AnchoredText('c', prop=dict(size=12), frameon=True, loc=2)
ax2.add_artist(at)


# ax1.scatter(, s=10, c='r', marker="o", label='second')
# ax1.scatter(, s=10, c='r', marker="o", label='second')
# plt.legend(loc='upper left');
plt.savefig(os.path.join(marcos_data, 'correlation.png'),
                  bbox_inches='tight')
