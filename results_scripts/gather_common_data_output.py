import os
import sys
import glob
import pandas as pd
from configobj import ConfigObj
from functools import reduce

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

input_data = os.path.join(MAIN_PATH, config['volume_results'])

# Where we stored merged data
output_path= os.path.join(MAIN_PATH, 'output_data/13_Merged_data')
if not os.path.exists(output_path):
    os.makedirs(output_path)

config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

exp_name = []
all_files = []
for exp in config_paths.config_path:
    name = exp.split("/")[-1]
    exp_path = os.path.join(input_data, exp)
    exp_name.append(name)
    all_files.append(glob.glob(exp_path + "/glacier_*.csv")[0])

print(all_files)

li = []
for filename, name in zip(all_files, exp_name):
    df = pd.read_csv(filename)
    li.append(df)

# Now lets find the glaciers for which we have data among all methods
df_common = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'), li)
df_common.to_csv(os.path.join(output_path,
                           'common_glaciers_all_methods.csv'))

# Selecting only the ice Cap
df_common_ice_cap = df_common[df_common['rgi_id'].str.match('RGI60-05.10315')].copy()

df_common_ice_cap.to_csv(os.path.join(output_path,
                           'common_ice_cap_all_methods.csv'))

# Removing the Ice cap!! from the experiments results
df_common_rest = df_common[~df_common.rgi_id.str.contains('RGI60-05.10315')]
df_common_rest.to_csv(os.path.join(output_path,
                           'common_glaciers_rest_all_methods.csv'))

# Read Farinotti and Huss data
df_consensus = pd.read_hdf(os.path.join(MAIN_PATH,
                                        config['consensus_data']))
df_consensus.reset_index(level=0, inplace=True)

df_consensus = df_consensus[['RGIId',
                             'Area',
                             'vol_itmix_m3',
                             'vol_bsl_itmix_m3',
                             'vol_model1_m3',
                             'vol_bsl_model1_m3'
                             ]]

df_consensus.reset_index(level=0, inplace=True)
df_consensus.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

# Convert units in km3
F_vol_km3 = df_consensus.loc[:, 'vol_itmix_m3'].copy() * 1e-9
F_vol_bsl_km3 = df_consensus.loc[:, 'vol_bsl_itmix_m3'].copy() * 1e-9
H_vol_km3 = df_consensus.loc[:, 'vol_model1_m3'].copy() * 1e-9
H_vol_bsl_km3 = df_consensus.loc[:, 'vol_bsl_model1_m3'].copy() * 1e-9

df_consensus.loc[:, 'vol_itmix_km3'] = F_vol_km3
df_consensus.loc[:, 'vol_bsl_itmix_km3'] = F_vol_bsl_km3
df_consensus.loc[:, 'Huss_vol_km3'] = H_vol_km3
df_consensus.loc[:, 'Huss_vol_bsl_km3'] = H_vol_bsl_km3

# Lets sort out first the Ice cap!!
# Selecting only the ice Cap
consensus_ice_cap = df_consensus[df_consensus['rgi_id'].str.match('RGI60-05.10315')].copy()

# Read in Ice cap preprocessing
df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))
print('Total ice cap basins')
print(len(df_prepro_ic))

# Exclude glaciers with prepro-erros from df_prepro_ic
de = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in df_prepro_ic.rgi_id]
df_prepro_ic = df_prepro_ic.iloc[keep_errors]

# Let just take data that we need
df_prepro_ic_to_save = df_prepro_ic[['rgi_id',
                                     'rgi_area_km2',
                                     'terminus_type',
                                     'inv_volume_km3',
                                     'volume_bsl']].set_index('rgi_id')

df_common_ice_cap_sel = df_common_ice_cap[['rgi_id',
                                           'rgi_area_km2_x',
                                           'k_itslive_lowbound_inv_volume_km3',
                                           'k_itslive_lowbound_volume_bsl',
                                           'k_itslive_value_inv_volume_km3',
                                           'k_itslive_value_volume_bsl',
                                           'k_itslive_upbound_inv_volume_km3',
                                           'k_itslive_upbound_volume_bsl',
                                           'k_measures_lowbound_inv_volume_km3',
                                           'k_measures_lowbound_volume_bsl',
                                           'k_measures_value_inv_volume_km3',
                                           'k_measures_value_volume_bsl',
                                           'k_measures_upbound_inv_volume_km3',
                                           'k_measures_upbound_volume_bsl',
                                           'k_racmo_lowbound_inv_volume_km3',
                                           'k_racmo_lowbound_volume_bsl',
                                           'k_racmo_value_inv_volume_km3',
                                           'k_racmo_value_volume_bsl',
                                           'k_racmo_upbound_inv_volume_km3',
                                           'k_racmo_upbound_volume_bsl']]

result_ic = pd.concat([df_prepro_ic_to_save,
                       df_common_ice_cap_sel.set_index('rgi_id'),
                       consensus_ice_cap.set_index('rgi_id')],
                      axis=1)
result_ic.index.name = 'rgi_id'

# result_ic.to_csv(os.path.join(output_path,
#                               'ice_cap_volume_all_methos_plus_consensus.csv'))

print('Total of the ice cap modelled')
print(len(result_ic))

# Selecting now for the rest of the glaciers that have data in all methods the
# volume from the consensus and also the volume before calving from
# the pre-processing run
print('This many glaciers we need at the end')
print(len(df_common_rest))

result_rest = pd.merge(left=df_common_rest,
                       right=df_consensus,
                       how='inner',
                       left_on = 'rgi_id',
                       right_on='rgi_id')

volume_df_rest = result_rest[['rgi_id',
                              'rgi_area_km2_x',
                              'k_itslive_lowbound_inv_volume_km3',
                              'k_itslive_lowbound_volume_bsl',
                              'k_itslive_value_inv_volume_km3',
                              'k_itslive_value_volume_bsl',
                              'k_itslive_upbound_inv_volume_km3',
                              'k_itslive_upbound_volume_bsl',
                              'k_measures_lowbound_inv_volume_km3',
                              'k_measures_lowbound_volume_bsl',
                              'k_measures_value_inv_volume_km3',
                              'k_measures_value_volume_bsl',
                              'k_measures_upbound_inv_volume_km3',
                              'k_measures_upbound_volume_bsl',
                              'k_racmo_lowbound_inv_volume_km3',
                              'k_racmo_lowbound_volume_bsl',
                              'k_racmo_value_inv_volume_km3',
                              'k_racmo_value_volume_bsl',
                              'k_racmo_upbound_inv_volume_km3',
                              'k_racmo_upbound_volume_bsl',
                              'Area',
                              'vol_itmix_m3',
                              'vol_bsl_itmix_m3',
                              'vol_model1_m3',
                              'vol_bsl_model1_m3',
                              'vol_itmix_km3',
                              'vol_bsl_itmix_km3',
                              'Huss_vol_km3',
                              'Huss_vol_bsl_km3'
]]

# Read in Preprocessing for all glaciers
df_prepro = pd.read_csv(os.path.join(MAIN_PATH,
                                     config['prepro_no_calving']))
df_prepro = df_prepro[['rgi_id',
                       'rgi_area_km2',
                       'glacier_type',
                       'terminus_type',
                       'status',
                       'inv_volume_km3',
                       'volume_bsl']]

result = pd.merge(left=volume_df_rest,
                  right=df_prepro,
                  how='inner',
                  left_on = 'rgi_id',
                  right_on='rgi_id')

result.to_csv(os.path.join(output_path,
                           'volume_rest_all_methos_plus_consensus.csv'))

print(len(result))


fa_df_rest = df_common_rest[['rgi_id',
                             'rgi_area_km2_x',
                             'glacier_type_x',
                             'terminus_type_x',
                             'k_itslive_lowbound_calving_flux',
                             'k_itslive_value_calving_flux',
                             'k_itslive_upbound_calving_flux',
                             'k_measures_lowbound_calving_flux',
                             'k_measures_value_calving_flux',
                             'k_measures_upbound_calving_flux',
                             'k_racmo_lowbound_calving_flux',
                             'k_racmo_value_calving_flux',
                             'k_racmo_upbound_calving_flux'
]]

fa_df_rest.to_csv(os.path.join(output_path, 'fa_results_common_methods_rest.csv'))

fa_df_ic = df_common_ice_cap[['rgi_id',
                              'rgi_area_km2_x',
                              'glacier_type_x',
                              'terminus_type_x',
                              'k_itslive_lowbound_calving_flux',
                              'k_itslive_value_calving_flux',
                              'k_itslive_upbound_calving_flux',
                              'k_measures_lowbound_calving_flux',
                              'k_measures_value_calving_flux',
                              'k_measures_upbound_calving_flux',
                              'k_racmo_lowbound_calving_flux',
                              'k_racmo_value_calving_flux',
                              'k_racmo_upbound_calving_flux'
]]

fa_df_ic.to_csv(os.path.join(output_path,
                             'fa_results_common_methods_ice_cap.csv'))