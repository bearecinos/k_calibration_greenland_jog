# Calculate total glacier volume for final plot
import os
import sys
import pandas as pd
import geopandas as gpd
import salem
from configobj import ConfigObj

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Were to store merged data
output_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test')

# Reading glacier directories per experiment
configurations_order = ['farinotti',
                        'huss',
                        'no_calving',
                        'k_measures_lowbound',
                        'k_measures_value',
                        'k_measures_upbound',
                        'k_itslive_lowbound',
                        'k_itslive_value',
                        'k_itslive_upbound',
                        'k_racmo_lowbound',
                        'k_racmo_value',
                        'k_racmo_upbound']

print(configurations_order)

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# Calculate study area
study_area = misc.get_study_area(rgidf, MAIN_PATH, config['ice_cap_prepro'])
print('Our target study area')
print(study_area)

# Read common glaciers data frame and select the data that we need
df_common = pd.read_csv(os.path.join(output_path,
                                     'common_final_results.csv'))

print('Originally we have this many glaciers '
      'in common among the different methods')
print(len(df_common))

# Selecting only the ice Cap
df_common_ice_cap = df_common[df_common['rgi_id'].str.match('RGI60-05.10315')].copy()
print('How many basis of the Ice cap can we solve with all methods')
print(len(df_common_ice_cap))

# Removing the Ice cap!! from the experiments results
df_common_rest = df_common[~df_common.rgi_id.str.contains('RGI60-05.10315')]
print('The amount of glaciers now without the ice cap')
print(len(df_common_rest))

# Lets separate the data between rest, ice cap
# and volume and volume below sea level
df_common_volume_glaciers = df_common_rest[['rgi_id',
                                            'rgi_area_km2',
                                            'volume_before_calving',
                                            'inv_volume_km3_k_measures_lowbound',
                                            'inv_volume_km3_k_measures_value',
                                            'inv_volume_km3_k_measures_upbound',
                                            'inv_volume_km3_k_itslive_lowbound',
                                            'inv_volume_km3_k_itslive_value',
                                            'inv_volume_km3_k_itslive_upbound',
                                            'inv_volume_km3_k_racmo_lowbound',
                                            'inv_volume_km3_k_racmo_value',
                                            'inv_volume_km3_k_racmo_upbound']]

df_common_volume_bsl_glaciers = df_common_rest[['rgi_id',
                                                'vbsl_k_measures_lowbound',
                                                'vbsl_c_k_measures_lowbound',
                                                'vbsl_c_k_measures_value',
                                                'vbsl_c_k_measures_upbound',
                                                'vbsl_c_k_itslive_lowbound',
                                                'vbsl_c_k_itslive_value',
                                                'vbsl_c_k_itslive_upbound',
                                                'vbsl_c_k_racmo_lowbound',
                                                'vbsl_c_k_racmo_value',
                                                'vbsl_c_k_racmo_upbound']]

# Lets pick one configuration for the volume below sea level without calving
# this will be the same for all the methods
df_common_volume_bsl_glaciers.rename(columns={'vbsl_k_measures_lowbound': 'vbsl_no_calving'}, inplace=True)

# Repeat the same for the ice cap
df_common_volume_ice_cap = df_common_ice_cap[['rgi_id',
                                              'rgi_area_km2',
                                              'volume_before_calving',
                                              'inv_volume_km3_k_measures_lowbound',
                                              'inv_volume_km3_k_measures_value',
                                              'inv_volume_km3_k_measures_upbound',
                                              'inv_volume_km3_k_itslive_lowbound',
                                              'inv_volume_km3_k_itslive_value',
                                              'inv_volume_km3_k_itslive_upbound',
                                              'inv_volume_km3_k_racmo_lowbound',
                                              'inv_volume_km3_k_racmo_value',
                                              'inv_volume_km3_k_racmo_upbound']]

df_common_volume_bsl_ice_cap = df_common_ice_cap[['rgi_id',
                                                  'vbsl_k_measures_lowbound',
                                                  'vbsl_c_k_measures_lowbound',
                                                  'vbsl_c_k_measures_value',
                                                  'vbsl_c_k_measures_upbound',
                                                  'vbsl_c_k_itslive_lowbound',
                                                  'vbsl_c_k_itslive_value',
                                                  'vbsl_c_k_itslive_upbound',
                                                  'vbsl_c_k_racmo_lowbound',
                                                  'vbsl_c_k_racmo_value',
                                                  'vbsl_c_k_racmo_upbound']]

# Lets pick one configuration for the volume below sea level without calving
# this will be the same for all the methods
df_common_volume_bsl_ice_cap.rename(columns={'vbsl_k_measures_lowbound': 'vbsl_no_calving'},
                                     inplace=True)

# Read Farinotti and Huss data
df_consensus = pd.read_hdf(os.path.join(MAIN_PATH,
                                        config['consensus_data']))
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

# Selecting only the ice Cap
consensus_ice_cap = df_consensus[df_consensus['rgi_id'].str.match('RGI60-05.10315')].copy()
print('Data for the Ice cap from consensus')
print(consensus_ice_cap)
consensus_rest = df_consensus.loc[df_consensus['rgi_id'] != 'RGI60-05.10315']

# Selecting only common glaciers between the consensus and OGGM common
# output for the three methods
ids = df_common_rest.rgi_id.values
keep_ids = [(i in ids) for i in consensus_rest.rgi_id]
consensus_rest = consensus_rest.loc[keep_ids]
print('Select from the consensus the same No. of glaciers '
      'that we can model with the three methods')
print(len(consensus_rest))

# Separate the Consensus volume and volume below
# sea level for the common glaciers and for the Ice cap
consensus_volume_glaciers = consensus_rest[['rgi_id',
                                            'Area',
                                            'vol_itmix_km3',
                                            'Huss_vol_km3']]

consensus_volume_bsl_glaciers = consensus_rest[['rgi_id',
                                                'vol_bsl_itmix_km3',
                                                'Huss_vol_bsl_km3']]

consensus_volume_ice_cap = consensus_ice_cap[['rgi_id',
                                              'Area',
                                              'vol_itmix_km3',
                                              'Huss_vol_km3']]

consensus_volume_bsl_ice_cap = consensus_ice_cap[['rgi_id',
                                                  'vol_bsl_itmix_km3',
                                                  'Huss_vol_bsl_km3']]

# Merge the results from the consensus with OGGM model output
df_merge_volume_glaciers = pd.merge(left=consensus_volume_glaciers,
                             right=df_common_volume_glaciers,
                             how='inner',
                             left_on = 'rgi_id',
                             right_on='rgi_id')

print('List of columns names for the merge '
      'file (consensus + oggm) containing volume results of all glaciers')
print(df_merge_volume_glaciers.columns)
print(df_merge_volume_glaciers.shape)

df_merge_volume_bsl_glaciers = pd.merge(left=consensus_volume_bsl_glaciers,
                                        right=df_common_volume_bsl_glaciers,
                                        how='inner',
                                        left_on = 'rgi_id',
                                        right_on='rgi_id')

print('List of columns names for the merge '
      'file (consensus + oggm) containing volume bsl results of all glaciers')
print(df_merge_volume_bsl_glaciers.columns)
print(df_merge_volume_bsl_glaciers.shape)

print('Differences in Area between the consensus and OGGM')
print(df_merge_volume_glaciers.Area.sum())
print(df_merge_volume_glaciers.rgi_area_km2.sum())
print('No difference! ..........')

# Ice cap pre-processing in OGGM, volumes before calving
# the pre-processing runs
df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))
df_prepro_ic_vbsl = pd.read_csv(os.path.join(MAIN_PATH,
                    'output_data/02_Ice_cap_prepo/volume_below_sea_level.csv'))

# Get the volume for Tidewater basins modelled by OGGM
ice_cap_tw_no_calving = df_prepro_ic.loc[df_prepro_ic['terminus_type'] ==
                                         'Marine-terminating']

# Get volume below sea level for Tidewater basins
ice_cap_tw_no_calving_vbsl = df_prepro_ic_vbsl.loc[df_prepro_ic_vbsl['terminus_type'] ==
                                                   'Marine-terminating']

# We don't manage to model the entire ice cap so we separate what we don't model
# In order to add that volume/and volume bsl later to our results after calving
ids_to_add_separate = df_common_volume_ice_cap.rgi_id.values
print(ids_to_add_separate)
keep_tw_ids = [(i not in ids_to_add_separate) for i in ice_cap_tw_no_calving.rgi_id.values]

# volume of TW basins that we cant model
no_model_tw_basins = ice_cap_tw_no_calving.loc[keep_tw_ids]
no_model_tw_basins_vbsl = ice_cap_tw_no_calving_vbsl.loc[keep_tw_ids]
print('How many Tidewater basins we dont model but we need to '
      'keep their volume to add it later')
print(len(no_model_tw_basins))
print(len(no_model_tw_basins_vbsl))

# Select the Land terminating parts
ice_cap_land_no_calving = df_prepro_ic.loc[df_prepro_ic['terminus_type'] ==
                                           'Land-terminating']
print('This many land terminating basins')
print(len(ice_cap_land_no_calving))

# Adding up ICE CAP volumes individually for each MODEL
fari_ice_cap = consensus_volume_ice_cap.vol_itmix_km3.values[0]
huss_ice_cap = consensus_volume_ice_cap.Huss_vol_km3.values[0]
fari_ice_cap_vbsl = consensus_volume_bsl_ice_cap.vol_bsl_itmix_km3.values[0]
huss_ice_cap_vbsl = consensus_volume_bsl_ice_cap.Huss_vol_bsl_km3.values[0]
print('Ice cap consensus volume')
print(fari_ice_cap)
print(huss_ice_cap)
print('Ice cap consensus area')
print(consensus_volume_ice_cap.Area.values[0])

# Adding OGGM part
no_calving_tw_ice_cap = ice_cap_tw_no_calving.inv_volume_km3.sum()
no_calving_land_ice_cap = ice_cap_land_no_calving.inv_volume_km3.sum()

# Add up the non model tw basins volume and volume bsl to add to results later
no_model_tw_basins_volume_no_calving = no_model_tw_basins.inv_volume_km3.sum()
no_model_tw_basins_vbsl_no_calving = no_model_tw_basins_vbsl['volume bsl'].sum()

oggm_ice_cap_no_calving = no_calving_tw_ice_cap + no_calving_land_ice_cap
oggm_ice_cap_no_calving_vbsl = df_prepro_ic_vbsl['volume bsl'].sum()

print('Ice cap volume OGGM no calving')
print(oggm_ice_cap_no_calving)
print('Ice cap OGGM area')
print(df_prepro_ic.rgi_area_km2.sum())

# Adding up the ICE CAP volume and volume bsl for each model configuration
print('Now we will add the volumes for this many basins')
print(len(df_common_volume_ice_cap))
print(len(no_model_tw_basins))

k_measures_lowbound_ice_cap = no_calving_land_ice_cap + \
                              no_model_tw_basins_volume_no_calving + \
                              df_common_volume_ice_cap.inv_volume_km3_k_measures_lowbound.sum()

k_measures_value_ice_cap =  no_calving_land_ice_cap + \
                            no_model_tw_basins_volume_no_calving + \
                            df_common_volume_ice_cap.inv_volume_km3_k_measures_value.sum()

k_measures_upbound_ice_cap = no_calving_land_ice_cap + \
                             no_model_tw_basins_volume_no_calving + \
                             df_common_volume_ice_cap.inv_volume_km3_k_measures_upbound.sum()

k_itslive_lowbound_ice_cap = no_calving_land_ice_cap + \
                             no_model_tw_basins_volume_no_calving + \
                             df_common_volume_ice_cap.inv_volume_km3_k_itslive_lowbound.sum()

k_itslive_value_ice_cap = no_calving_land_ice_cap + \
                          no_model_tw_basins_volume_no_calving + \
                          df_common_volume_ice_cap.inv_volume_km3_k_itslive_value.sum()

k_itslive_upbound_ice_cap = no_calving_land_ice_cap + \
                            no_model_tw_basins_volume_no_calving + \
                            df_common_volume_ice_cap.inv_volume_km3_k_itslive_upbound.sum()


k_racmo_lowbound_ice_cap = no_calving_land_ice_cap + \
                           no_model_tw_basins_volume_no_calving + \
                           df_common_volume_ice_cap.inv_volume_km3_k_racmo_lowbound.sum()

k_racmo_value_ice_cap = no_calving_land_ice_cap + \
                        no_model_tw_basins_volume_no_calving +\
                        df_common_volume_ice_cap.inv_volume_km3_k_racmo_value.sum()

k_racmo_upbound_ice_cap = no_calving_land_ice_cap + \
                          no_model_tw_basins_volume_no_calving + \
                          df_common_volume_ice_cap.inv_volume_km3_k_racmo_upbound.sum()

# Ice cap volume below sea level
print('Check if we have 9 entities for adding the volume bsl')
print(len(no_model_tw_basins_vbsl)+ len(df_common_volume_bsl_ice_cap))

k_measures_lowbound_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving + \
                                   df_common_volume_bsl_ice_cap.vbsl_c_k_measures_lowbound.sum()

k_measures_value_ice_cap_vbsl =  no_model_tw_basins_vbsl_no_calving +\
                                 df_common_volume_bsl_ice_cap.vbsl_c_k_measures_value.sum()

k_measures_upbound_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                                  df_common_volume_bsl_ice_cap.vbsl_c_k_measures_upbound.sum()

k_itslive_lowbound_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                             df_common_volume_bsl_ice_cap.vbsl_c_k_itslive_lowbound.sum()

k_itslive_value_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                          df_common_volume_bsl_ice_cap.vbsl_c_k_itslive_value.sum()

k_itslive_upbound_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                            df_common_volume_bsl_ice_cap.vbsl_c_k_itslive_upbound.sum()

k_racmo_lowbound_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                           df_common_volume_bsl_ice_cap.vbsl_c_k_racmo_lowbound.sum()

k_racmo_value_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                        df_common_volume_bsl_ice_cap.vbsl_c_k_racmo_value.sum()

k_racmo_upbound_ice_cap_vbsl = no_model_tw_basins_vbsl_no_calving +\
                          df_common_volume_bsl_ice_cap.vbsl_c_k_racmo_upbound.sum()

# Building the data arrays and total volume data frame
# FIRST FOR ALL GLACIERS
Areas = [df_merge_volume_glaciers.Area.sum() + consensus_volume_ice_cap.Area.values[0],
         df_merge_volume_glaciers.Area.sum() + consensus_volume_ice_cap.Area.values[0],
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum(),
         df_merge_volume_glaciers.rgi_area_km2.sum() + df_prepro_ic.rgi_area_km2.sum()]

Glaciers_total_volume = [df_merge_volume_glaciers.vol_itmix_km3.sum(),
                         df_merge_volume_glaciers.Huss_vol_km3.sum(),
                         df_merge_volume_glaciers.volume_before_calving.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_measures_lowbound.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_measures_value.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_measures_upbound.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_itslive_lowbound.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_itslive_value.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_itslive_upbound.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_racmo_lowbound.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_racmo_value.sum(),
                         df_merge_volume_glaciers.inv_volume_km3_k_racmo_upbound.sum()]

Glaciers_total_volume_bsl = [df_merge_volume_bsl_glaciers.vol_bsl_itmix_km3.sum(),
                             df_merge_volume_bsl_glaciers.Huss_vol_bsl_km3.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_no_calving.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_measures_lowbound.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_measures_value.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_measures_upbound.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_itslive_lowbound.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_itslive_value.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_itslive_upbound.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_racmo_lowbound.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_racmo_value.sum(),
                             df_merge_volume_bsl_glaciers.vbsl_c_k_racmo_upbound.sum()]

Ice_cap_total_volume = [fari_ice_cap,
                        huss_ice_cap,
                        oggm_ice_cap_no_calving,
                        k_measures_lowbound_ice_cap,
                        k_measures_value_ice_cap,
                        k_measures_upbound_ice_cap,
                        k_itslive_lowbound_ice_cap,
                        k_itslive_value_ice_cap,
                        k_itslive_upbound_ice_cap,
                        k_racmo_lowbound_ice_cap,
                        k_racmo_value_ice_cap,
                        k_racmo_upbound_ice_cap]

Ice_cap_total_volume_bsl = [fari_ice_cap_vbsl,
                            huss_ice_cap_vbsl,
                            oggm_ice_cap_no_calving_vbsl,
                            k_measures_lowbound_ice_cap_vbsl,
                            k_measures_value_ice_cap_vbsl,
                            k_measures_upbound_ice_cap_vbsl,
                            k_itslive_lowbound_ice_cap_vbsl,
                            k_itslive_value_ice_cap_vbsl,
                            k_itslive_upbound_ice_cap_vbsl,
                            k_racmo_lowbound_ice_cap_vbsl,
                            k_racmo_value_ice_cap_vbsl,
                            k_racmo_upbound_ice_cap_vbsl]


df_volumes_total = {'Configuration': configurations_order,
                    'Area': Areas,
                    'Volume_all_glaciers': Glaciers_total_volume,
                    'Volume_ice_cap': Ice_cap_total_volume,
                    'Volume_all_glaciers_bsl': Glaciers_total_volume_bsl,
                    'Volume_ice_cap_bsl': Ice_cap_total_volume_bsl}

data_frame = pd.DataFrame(data=df_volumes_total)

# data_frame.to_csv(os.path.join(output_path +
#                                    '/total_volume_vbsl_for_final_plot.csv'))