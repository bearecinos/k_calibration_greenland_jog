import os
import sys
import numpy as np
from configobj import ConfigObj
from collections import defaultdict
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import glob
import geopandas as gpd
import salem
from scipy import stats
from oggm import utils
import argparse

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf

config = ConfigObj(os.path.expanduser(config_file))

MAIN_PATH = config['main_repo_path']
sys.path.append(MAIN_PATH)

from k_tools import misc

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

input_data_path = config['input_data_folder']
main_output = os.path.join(MAIN_PATH, 'output_data')

enderlin_path = os.path.join(input_data_path, 'geo_data_5_enderlin_08022021/enderlin.shp')
enderlin_data = gpd.read_file(enderlin_path)

enderlin_data.rename(columns={'commune': 'rgi_id'}, inplace=True)
obs_f = pd.DataFrame(enderlin_data.drop(columns='geometry'))

obs_f_error = pd.read_csv(os.path.join(input_data_path,
                                       'geo_data_5_enderlin_08022021/GreenlandGIC_discharge-uncertainty_timeseries.csv'))

obs_fa_full = pd.merge(obs_f, obs_f_error, how="left", on='BoxID')

long_term_avg = obs_fa_full[['rgi_id']].copy()

long_term_avg['long term median m^3/yr'] = obs_fa_full[['1985_x', '1986_x', '1987_x', '1988_x', '1989_x',
                                                        '1990_x', '1991_x', '1992_x', '1993_x', '1994_x', '1995_x', '1996_x', '1997_x', '1998_x', '1999_x',
                                                        '2000_x', '2001_x', '2002_x', '2003_x', '2004_x', '2005_x', '2006_x', '2007_x', '2008_x', '2009_x',
                                                        '2010_x', '2011_x', '2012_x', '2013_x', '2014_x', '2015_x', '2016_x', '2017_x', '2018_x']].median(axis=1)

long_term_avg['long term mean m^3/yr'] = obs_fa_full[['1985_x', '1986_x', '1987_x', '1988_x', '1989_x',
                                                      '1990_x', '1991_x', '1992_x', '1993_x', '1994_x', '1995_x', '1996_x', '1997_x', '1998_x', '1999_x',
                                                      '2000_x', '2001_x', '2002_x', '2003_x', '2004_x', '2005_x', '2006_x', '2007_x', '2008_x', '2009_x',
                                                      '2010_x', '2011_x', '2012_x', '2013_x', '2014_x', '2015_x', '2016_x', '2017_x', '2018_x']].mean(axis=1)

long_term_avg['long term STD m^3/yr'] =  obs_fa_full[['1985_x', '1986_x', '1987_x', '1988_x', '1989_x', '1990_x',
                                               '1991_x', '1992_x', '1993_x', '1994_x', '1995_x', '1996_x', '1997_x', '1998_x', '1999_x',
                                               '2000_x', '2001_x', '2002_x', '2003_x', '2004_x', '2005_x', '2006_x', '2007_x', '2008_x', '2009_x',
                                               '2010_x', '2011_x', '2012_x', '2013_x', '2014_x', '2015_x', '2016_x', '2017_x', '2018_x']].std(axis=1)

long_term_avg['long term error MAD m^3/yr'] =  obs_fa_full[['1985_y', '1986_y', '1987_y', '1988_y', '1989_y', '1990_y',
                                               '1991_y', '1992_y', '1993_y', '1994_y', '1995_y', '1996_y', '1997_y', '1998_y', '1999_y',
                                               '2000_y', '2001_y', '2002_y', '2003_y', '2004_y', '2005_y', '2006_y', '2007_y', '2008_y', '2009_y',
                                               '2010_y', '2011_y', '2012_y', '2013_y', '2014_y', '2015_y', '2016_y', '2017_y', '2018_y']].mad(axis=1)

long_term_avg['period one median m^3/yr'] = obs_fa_full[['1985_x', '1986_x', '1987_x', '1988_x', '1989_x', '1990_x',
                                               '1991_x', '1992_x', '1993_x', '1994_x', '1995_x', '1996_x', '1997_x', '1998_x']].median(axis=1)

long_term_avg['period one mean m^3/yr'] = obs_fa_full[['1985_x', '1986_x', '1987_x', '1988_x', '1989_x', '1990_x',
                                               '1991_x', '1992_x', '1993_x', '1994_x', '1995_x', '1996_x', '1997_x', '1998_x']].mean(axis=1)


long_term_avg['period one STD m^3/yr'] =  obs_fa_full[['1985_x', '1986_x', '1987_x', '1988_x', '1989_x', '1990_x',
                                               '1991_x', '1992_x', '1993_x', '1994_x', '1995_x', '1996_x', '1997_x', '1998_x']].std(axis=1)

long_term_avg['period one error MAD m^3/yr'] =  obs_fa_full[['1985_y', '1986_y', '1987_y', '1988_y', '1989_y', '1990_y',
                                               '1991_y', '1992_y', '1993_y', '1994_y', '1995_y', '1996_y', '1997_y', '1998_y']].mad(axis=1)

long_term_avg['period two median m^3/yr'] = obs_fa_full[['1999_x',
                                                  '2000_x', '2001_x', '2002_x', '2003_x', '2004_x', '2005_x', '2006_x', '2007_x', '2008_x', '2009_x',
                                                  '2010_x', '2011_x', '2012_x', '2013_x', '2014_x', '2015_x', '2016_x', '2017_x', '2018_x']].median(axis=1)

long_term_avg['period two mean m^3/yr'] = obs_fa_full[['1999_x',
                                                  '2000_x', '2001_x', '2002_x', '2003_x', '2004_x', '2005_x', '2006_x', '2007_x', '2008_x', '2009_x',
                                                  '2010_x', '2011_x', '2012_x', '2013_x', '2014_x', '2015_x', '2016_x', '2017_x', '2018_x']].mean(axis=1)

long_term_avg['period two STD m^3/yr'] =  obs_fa_full[['1999_x',
                                                       '2000_x', '2001_x', '2002_x', '2003_x', '2004_x', '2005_x', '2006_x', '2007_x', '2008_x', '2009_x',
                                                       '2010_x', '2011_x', '2012_x', '2013_x', '2014_x', '2015_x', '2016_x', '2017_x', '2018_x']].std(axis=1)

long_term_avg['period two error MAD m^3/yr'] =  obs_fa_full[['1999_y',
                                                         '2000_y', '2001_y', '2002_y', '2003_y', '2004_y', '2005_y', '2006_y', '2007_y', '2008_y', '2009_y',
                                                         '2010_y', '2011_y', '2012_y', '2013_y', '2014_y', '2015_y', '2016_y', '2017_y', '2018_y']].mad(axis=1)

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

d_fa = defaultdict(list)

for f, name in zip(all_files, exp_name):
    print(name)
    d_fa[name] = misc.combine_fa_data(f, long_term_avg, name)

correlations_exp_names = ['itslive', 'measures', 'racmo']

for exp in correlations_exp_names:
    exp_df = d_fa['k_'+exp+'_value']
    print(exp)
    print('Pearson------------------------------------')
    print('Long term avg', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['long term mean m^3/yr']/(1000**3)))
    print('Period one', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['period one mean m^3/yr']/(1000**3)))
    print('Period two', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['period two mean m^3/yr']/(1000**3)))
    print('sperman---------------------------------')
    print('Long term avg', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['long term mean m^3/yr']/(1000**3), method='spearman'))
    print('Period one', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['period one mean m^3/yr']/(1000**3), method='spearman'))
    print('Period two', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['period two mean m^3/yr']/(1000**3), method='spearman'))
    print('kendall----------------------------------')
    print('Long term avg', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['long term mean m^3/yr']/(1000**3), method='kendall'))
    print('Period one', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['period one mean m^3/yr']/(1000**3), method='kendall'))
    print('Period two', exp_df['k_'+exp+'_value_calving_flux'].corr(exp_df['period two mean m^3/yr']/(1000**3), method='kendall'))

periods = np.repeat(['long term mean', '1985-1998 mean', '1999-2018 mean'], 3)
experiments = np.repeat(correlations_exp_names, 3)

study_area = 32202.540

r_coff_pearson = []
p_values_pearson = []
r_coff_spearman = []
p_values_spearman = []

d_test = defaultdict(list)

for exp in correlations_exp_names:
    exp_df = d_fa['k_' + exp + '_value']
    exp_df['rgi_area_km2'] = d_fa['k_' + exp + '_lowbound']['rgi_area_km2']
    print(exp)
    for period in ['long term mean m^3/yr', 'period one mean m^3/yr', 'period two mean m^3/yr']:
        exp_df_p = exp_df.dropna(subset=[period])

        print('Period', period)
        print('Number of glaciers', len(exp_df_p))
        print('Area coverage', exp_df_p['rgi_area_km2'].sum())
        print('% Area coverage', exp_df_p['rgi_area_km2'].sum() * 100 / study_area)

        out_pearson = stats.pearsonr(exp_df_p['k_' + exp + '_value_calving_flux'].values,
                                     exp_df_p[period] / (1000 ** 3))
        out_spearman = stats.spearmanr(exp_df_p['k_' + exp + '_value_calving_flux'].values,
                                       exp_df_p[period] / (1000 ** 3))

        r_coff_pearson = np.append(r_coff_pearson, out_pearson[0])
        p_values_pearson = np.append(p_values_pearson, out_pearson[1])

        r_coff_spearman = np.append(r_coff_spearman, out_spearman[0])
        p_values_spearman = np.append(p_values_spearman, out_spearman[1])

        z_M = np.arange(0, len(exp_df_p), 1)

        out = misc.calculate_statistics(exp_df_p[period] / (1000 ** 3),
                                        exp_df_p['k_' + exp + '_value_calving_flux'],
                                        exp_df_p['rgi_area_km2'].sum() * 100 / study_area,
                                        z_M)

        d = {'test': out[0], 'zline': out[1], 'wline': out[2]}
        d_test[exp + '__' + period] = d

        print(out_pearson)
        print(out_spearman)
        print('---------------------------------')

# PARAMS for plots
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
sns.set_context('poster')

color_palette_k = sns.color_palette("muted")

from mpl_toolkits.axes_grid1 import make_axes_locatable
r = 1.2

fig1 = plt.figure(figsize=(14*r, 10*r))#, constrained_layout=True)
spec = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.3)

ax0 = plt.subplot(spec[0])
########## Data for itslive #############################
df_itslive = d_fa['k_itslive_value']
df_itslive['k_itslive_lowbound_calving_flux'] = d_fa['k_itslive_lowbound'].k_itslive_lowbound_calving_flux
df_itslive['k_itslive_upbound_calving_flux'] = d_fa['k_itslive_upbound'].k_itslive_upbound_calving_flux
################################
period = 'long term mean m^3/yr'
period_error = 'long term STD m^3/yr'
df_to_plot = df_itslive.dropna(subset=[period])
z_M = np.arange(0, len(df_to_plot), 1)
fa_oggm_itslive = df_to_plot['k_itslive_value_calving_flux']
fa_oggm_itslive_error = df_to_plot['k_itslive_upbound_calving_flux']-df_to_plot['k_itslive_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax0.errorbar(fa_enderlin, fa_oggm_itslive,
             xerr=fa_enderlin_std, yerr=fa_oggm_itslive_error,
             color=sns.xkcd_rgb["light green"], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1985-2018')
################################
period = 'period one mean m^3/yr'
period_error = 'period one STD m^3/yr'
df_to_plot = df_itslive.dropna(subset=[period])
z_M_2 = np.arange(0, len(df_to_plot), 1)
fa_oggm_itslive = df_to_plot['k_itslive_value_calving_flux']
fa_oggm_itslive_error = df_to_plot['k_itslive_upbound_calving_flux']-df_to_plot['k_itslive_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax0.errorbar(fa_enderlin, fa_oggm_itslive,
             xerr=fa_enderlin_std, yerr=fa_oggm_itslive_error,
             color=sns.xkcd_rgb["green"], fmt='*', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1985-1998')
#################################
period = 'period two mean m^3/yr'
period_error = 'period two STD m^3/yr'
df_to_plot = df_itslive.dropna(subset=[period])
z_M_3 = np.arange(0, len(df_to_plot), 1)
fa_oggm_itslive = df_to_plot['k_itslive_value_calving_flux']
fa_oggm_itslive_error = df_to_plot['k_itslive_upbound_calving_flux']-df_to_plot['k_itslive_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax0.errorbar(fa_enderlin, fa_oggm_itslive,
             xerr=fa_enderlin_std, yerr=fa_oggm_itslive_error,
             color=sns.xkcd_rgb["green"], fmt='s', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1999-2018')
##################################
ax0.plot(z_M, d_test['itslive__long term mean m^3/yr']['zline'], color=sns.xkcd_rgb["light green"], label='1985-2018')
ax0.plot(z_M_2, d_test['itslive__period one mean m^3/yr']['zline'], color=sns.xkcd_rgb["green"], label='1985-1998')
ax0.plot(z_M_3, d_test['itslive__period two mean m^3/yr']['zline'], color=sns.xkcd_rgb["dark green"], alpha=0.7, label='1999-2018')
ax0.plot(z_M, d_test['itslive__long term mean m^3/yr']['wline'], color='grey')
ax0.set_xticks([0, 0.10, 0.20, 0.30])
ax0.set_xlim(-0.01, 0.3)
ax0.set_ylim(-0.01, 0.3)
ax0.legend(fontsize=14, loc=1)
ax0.set_xlabel('Frontal ablation flux \n [km$^3$.yr$^{-1}$] \n '
              '(Bollen et. al. 2022)')
ax0.set_ylabel('Frontal ablation flux \n [km$^3$.yr$^{-1}$]')
#ax0.add_artist(d_test['itslive__long term mean m^3/yr']['test'])
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc='upper left')
ax0.add_artist(at)

####################################################################################################################################
ax1 = plt.subplot(spec[1])
########## Data for measures #############################
df_measures = d_fa['k_measures_value']
df_measures['k_measures_lowbound_calving_flux'] = d_fa['k_measures_lowbound'].k_measures_lowbound_calving_flux
df_measures['k_measures_upbound_calving_flux'] = d_fa['k_measures_upbound'].k_measures_upbound_calving_flux
################################
period = 'long term mean m^3/yr'
period_error = 'long term STD m^3/yr'
df_to_plot = df_measures.dropna(subset=[period])
z_M = np.arange(0, len(df_to_plot), 1)
fa_oggm_measures = df_to_plot['k_measures_value_calving_flux']
fa_oggm_measures_error = df_to_plot['k_measures_upbound_calving_flux']-df_to_plot['k_measures_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax1.errorbar(fa_enderlin, fa_oggm_measures,
             xerr=fa_enderlin_std, yerr=fa_oggm_measures_error,
             color=sns.xkcd_rgb["light blue"], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1985-2018')
#############################################
period = 'period one mean m^3/yr'
period_error = 'period one STD m^3/yr'
df_to_plot = df_measures.dropna(subset=[period])
z_M_2 = np.arange(0, len(df_to_plot), 1)
fa_oggm_measures = df_to_plot['k_measures_value_calving_flux']
fa_oggm_measures_error = df_to_plot['k_measures_upbound_calving_flux']-df_to_plot['k_measures_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax1.errorbar(fa_enderlin, fa_oggm_measures,
             xerr=fa_enderlin_std, yerr=fa_oggm_measures_error,
             color=sns.xkcd_rgb["blue"], fmt='*', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1985-1998')
############################################
period = 'period two mean m^3/yr'
period_error = 'period two STD m^3/yr'
df_to_plot = df_measures.dropna(subset=[period])
z_M_3 = np.arange(0, len(df_to_plot), 1)
fa_oggm_measures = df_to_plot['k_measures_value_calving_flux']
fa_oggm_measures_error = df_to_plot['k_measures_upbound_calving_flux']-df_to_plot['k_measures_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax1.errorbar(fa_enderlin, fa_oggm_measures,
             xerr=fa_enderlin_std, yerr=fa_oggm_measures_error,
             color=sns.xkcd_rgb["dark blue"], fmt='s', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1999-2018')
###########################################
ax1.plot(z_M, d_test['measures__long term mean m^3/yr']['zline'], color=sns.xkcd_rgb["light blue"], label='1985-2018')
ax1.plot(z_M_2, d_test['measures__period one mean m^3/yr']['zline'], color=sns.xkcd_rgb["blue"], label='1985-1998')
ax1.plot(z_M_3, d_test['measures__period two mean m^3/yr']['zline'], color=sns.xkcd_rgb["dark blue"], label='1999-2018')
ax1.plot(z_M, d_test['measures__long term mean m^3/yr']['wline'], color='grey')
ax1.set_xticks([0, 0.10, 0.20, 0.30])
ax1.set_xlim(-0.01, 0.3)
ax1.set_ylim(-0.01, 0.3)
ax1.legend(fontsize=14, loc=1)
ax1.set_xlabel('Frontal ablation flux \n [km$^3$.yr$^{-1}$] \n '
              '(Bollen et. al. 2022)')
#ax1.add_artist(d_test['measures__long term mean m^3/yr']['test'])
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc='upper left')
ax1.add_artist(at)


######################################################################################################################################
ax2 = plt.subplot(spec[2])
### Getting data for plot racmo ######################################
df_racmo = d_fa['k_racmo_value']
df_racmo['k_racmo_lowbound_calving_flux'] = d_fa['k_racmo_lowbound'].k_racmo_lowbound_calving_flux
df_racmo['k_racmo_upbound_calving_flux'] = d_fa['k_racmo_upbound'].k_racmo_upbound_calving_flux
################################################################
period = 'long term mean m^3/yr'
period_error = 'long term STD m^3/yr'
df_to_plot = df_racmo.dropna(subset=[period])
z_M = np.arange(0, len(df_to_plot), 1)
fa_oggm_racmo = df_to_plot['k_racmo_value_calving_flux']
fa_oggm_racmo_error = df_to_plot['k_racmo_upbound_calving_flux']-df_to_plot['k_racmo_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax2.errorbar(fa_enderlin, fa_oggm_racmo,
             xerr=fa_enderlin_std, yerr=fa_oggm_racmo_error,
             color=sns.xkcd_rgb["light orange"], fmt='o', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1985-2018')
##############################################
period = 'period one mean m^3/yr'
period_error = 'period one STD m^3/yr'
df_to_plot = df_racmo.dropna(subset=[period])
z_M_2 = np.arange(0, len(df_to_plot), 1)
fa_oggm_racmo = df_to_plot['k_racmo_value_calving_flux']
fa_oggm_racmo_error = df_to_plot['k_racmo_upbound_calving_flux']-df_to_plot['k_racmo_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax2.errorbar(fa_enderlin, fa_oggm_racmo,
             xerr=fa_enderlin_std, yerr=fa_oggm_racmo_error,
             color=sns.xkcd_rgb["orange"], fmt='*', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1985-1998')
##########################################
period = 'period two mean m^3/yr'
period_error = 'period two STD m^3/yr'
df_to_plot = df_racmo.dropna(subset=[period])
z_M_3 = np.arange(0, len(df_to_plot), 1)
fa_oggm_racmo = df_to_plot['k_racmo_value_calving_flux']
fa_oggm_racmo_error = df_to_plot['k_racmo_upbound_calving_flux']-df_to_plot['k_racmo_lowbound_calving_flux']
fa_enderlin = df_to_plot[period]/(1000**3)
fa_enderlin_std = df_to_plot[period_error]/(1000**3)
ax2.errorbar(fa_enderlin, fa_oggm_racmo,
             xerr=fa_enderlin_std, yerr=fa_oggm_racmo_error,
             color=sns.xkcd_rgb["dark orange"], fmt='s', alpha=0.3,
             ecolor=sns.xkcd_rgb["dark grey"],
             elinewidth=0.5, label='1999-2018')
###########################################
ax2.plot(z_M, d_test['racmo__long term mean m^3/yr']['zline'], color=sns.xkcd_rgb["light orange"], label='1985-2018')
ax2.plot(z_M_2, d_test['racmo__period one mean m^3/yr']['zline'], color=sns.xkcd_rgb["orange"], label='1985-1998')
ax2.plot(z_M_3, d_test['racmo__period two mean m^3/yr']['zline'], color=sns.xkcd_rgb["dark orange"], alpha=0.3 , label='1999-2018')
ax2.plot(z_M, d_test['racmo__long term mean m^3/yr']['wline'], color='grey')
ax2.set_xticks([0, 0.10, 0.20, 0.30])
ax2.set_xlim(-0.01, 0.3)
ax2.set_ylim(-0.01, 0.3)
ax2.legend(fontsize=14, loc=1)
ax2.set_xlabel('Frontal ablation flux \n [km$^3$.yr$^{-1}$] \n '
              '(Bollen et. al. 2022)')
#ax2.add_artist(d_test['racmo__long term mean m^3/yr']['test'])
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc='upper left')
ax2.add_artist(at)


ax0.set_title('ITSLIVE')
ax1.set_title('MEaSUREs')
ax2.set_title('RACMO')

plt.savefig(os.path.join(plot_path,
                         'FA_correlations.png'),
            bbox_inches='tight', dpi=150)