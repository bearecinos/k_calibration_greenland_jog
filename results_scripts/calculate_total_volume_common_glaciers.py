# Calculate total glacier volume for final plot
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import salem
from configobj import ConfigObj

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Were to store merged data
output_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data')

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
df_vol_rest = pd.read_csv(os.path.join(output_path,
                                       'vol_results_common_methods_rest.csv'))
print('Number of glaciers modelled')
print(len(df_vol_rest))


# Read ice cap results
df_vol_ice_cap = pd.read_csv(os.path.join(output_path,
                                       'vol_results_common_methods_ice_cap.csv'))

print('Number of ice cap basins')
print(len(df_vol_ice_cap))

# Get the volume for Tidewater basins modelled by OGGM
ice_cap_tw = df_vol_ice_cap.loc[df_vol_ice_cap['terminus_type'] ==
                                           'Marine-terminating']

# Select the Land terminating parts
ice_cap_land = df_vol_ice_cap.loc[df_vol_ice_cap['terminus_type'] ==
                                  'Land-terminating']

# We don't manage to model the entire ice cap so we separate what we don't model
# In order to add that volume/and volume bsl later to our results after calving
modelled_ic = ['RGI60-05.10315_d350',
               'RGI60-05.10315_d14',
               'RGI60-05.10315_d336',
               'RGI60-05.10315_d12']

keep_tw_ids = [(i not in modelled_ic) for i in ice_cap_tw.rgi_id.values]
keep_model_ids = [(i in modelled_ic) for i in ice_cap_tw.rgi_id.values]

# volume of TW basins that we cant model
ice_cap_tw_no_modelled = ice_cap_tw[keep_tw_ids]
ice_cap_tw_modelled = ice_cap_tw[keep_model_ids]

print('How many Tidewater basins we dont model but we need to '
      'keep their volume to add it later')
print(len(ice_cap_tw_no_modelled))

print('This many land terminating basins')
print(len(ice_cap_land))

print('This are the basins that we modelled')
print(len(ice_cap_tw_modelled))

print(ice_cap_tw_modelled.columns.values)

# Adding OGGM part
no_calving_tw_ice_cap = ice_cap_tw_no_modelled.inv_volume_km3.sum() + ice_cap_tw_modelled.inv_volume_km3.sum()
no_calving_land_ice_cap = ice_cap_land.inv_volume_km3.sum()

oggm_ice_cap_no_calving = no_calving_tw_ice_cap + no_calving_land_ice_cap
print('Volume of the ice cap without calving')
print(oggm_ice_cap_no_calving)

# Now volume below sea level
no_calving_tw_ice_cap_bsl = ice_cap_tw_no_modelled.volume_bsl.sum() + ice_cap_tw_modelled.volume_bsl.sum()
no_calving_land_ice_cap_bsl = ice_cap_land.volume_bsl.sum()

oggm_ice_cap_no_calving_bsl = no_calving_tw_ice_cap_bsl + no_calving_land_ice_cap_bsl
print('Volume below sea level of the ice cap without calving')
print(oggm_ice_cap_no_calving_bsl)


##############################################################################
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
volume_ice_cap_Fari = consensus_ice_cap.vol_itmix_km3.values[0]
volume_bsl_ice_cap_Fari = consensus_ice_cap.vol_bsl_itmix_km3.values[0]

volume_ice_cap_Huss = consensus_ice_cap.Huss_vol_km3.values[0]
volume_bsl_ice_cap_Huss = consensus_ice_cap.Huss_vol_bsl_km3.values[0]

print('Farinotti ice cap vol')
print(volume_ice_cap_Fari)
print('Farinotti ice cap vol_bsl')
print(volume_bsl_ice_cap_Fari)

print('Huss ice cap vol')
print(volume_ice_cap_Huss)
print('Huss ice cap vol_bsl')
print(volume_bsl_ice_cap_Huss)

# Adding up the ICE CAP volume and volume bsl for each model configuration
print('Now we will add the volumes for this many basins')
print(len(ice_cap_land))
print(len(ice_cap_tw_no_modelled))
print(len(ice_cap_tw_modelled))
#
k_measures_lowbound_ice_cap = no_calving_land_ice_cap + \
                              ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                              ice_cap_tw_modelled.k_measures_lowbound_inv_volume_km3.sum()

k_measures_value_ice_cap =  no_calving_land_ice_cap + \
                            ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                            ice_cap_tw_modelled.k_measures_value_inv_volume_km3.sum()
#
k_measures_upbound_ice_cap = no_calving_land_ice_cap + \
                             ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                             ice_cap_tw_modelled.k_measures_upbound_inv_volume_km3.sum()

k_itslive_lowbound_ice_cap = no_calving_land_ice_cap + \
                              ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                              ice_cap_tw_modelled.k_itslive_lowbound_inv_volume_km3.sum()

k_itslive_value_ice_cap =  no_calving_land_ice_cap + \
                            ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                            ice_cap_tw_modelled.k_itslive_value_inv_volume_km3.sum()
#
k_itslive_upbound_ice_cap = no_calving_land_ice_cap + \
                             ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                             ice_cap_tw_modelled.k_itslive_upbound_inv_volume_km3.sum()

k_racmo_lowbound_ice_cap = no_calving_land_ice_cap + \
                              ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                              ice_cap_tw_modelled.k_racmo_lowbound_inv_volume_km3.sum()

k_racmo_value_ice_cap =  no_calving_land_ice_cap + \
                         ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                         ice_cap_tw_modelled.k_racmo_value_inv_volume_km3.sum()
#
k_racmo_upbound_ice_cap = no_calving_land_ice_cap + \
                          ice_cap_tw_no_modelled.inv_volume_km3.sum() + \
                          ice_cap_tw_modelled.k_racmo_upbound_inv_volume_km3.sum()

# Ice cap volume below sea level

k_measures_lowbound_ice_cap_vbsl = no_calving_land_ice_cap_bsl + \
                              ice_cap_tw_no_modelled.volume_bsl.sum() + \
                              ice_cap_tw_modelled.k_measures_lowbound_volume_bsl.sum()

k_measures_value_ice_cap_vbsl =  no_calving_land_ice_cap_bsl + \
                            ice_cap_tw_no_modelled.volume_bsl.sum() + \
                            ice_cap_tw_modelled.k_measures_value_volume_bsl.sum()
#
k_measures_upbound_ice_cap_vbsl = no_calving_land_ice_cap_bsl + \
                             ice_cap_tw_no_modelled.volume_bsl.sum() + \
                             ice_cap_tw_modelled.k_measures_upbound_volume_bsl.sum()

k_itslive_lowbound_ice_cap_vbsl = no_calving_land_ice_cap_bsl + \
                              ice_cap_tw_no_modelled.volume_bsl.sum() + \
                              ice_cap_tw_modelled.k_itslive_lowbound_volume_bsl.sum()

k_itslive_value_ice_cap_vbsl =  no_calving_land_ice_cap_bsl + \
                            ice_cap_tw_no_modelled.volume_bsl.sum() + \
                            ice_cap_tw_modelled.k_itslive_value_volume_bsl.sum()
#
k_itslive_upbound_ice_cap_vbsl = no_calving_land_ice_cap_bsl + \
                             ice_cap_tw_no_modelled.volume_bsl.sum() + \
                             ice_cap_tw_modelled.k_itslive_upbound_volume_bsl.sum()

k_racmo_lowbound_ice_cap_vbsl = no_calving_land_ice_cap_bsl + \
                              ice_cap_tw_no_modelled.volume_bsl.sum() + \
                              ice_cap_tw_modelled.k_racmo_lowbound_volume_bsl.sum()

k_racmo_value_ice_cap_vbsl =  no_calving_land_ice_cap_bsl + \
                         ice_cap_tw_no_modelled.volume_bsl.sum() + \
                         ice_cap_tw_modelled.k_racmo_value_volume_bsl.sum()
#
k_racmo_upbound_ice_cap_vbsl = no_calving_land_ice_cap_bsl + \
                          ice_cap_tw_no_modelled.volume_bsl.sum() + \
                          ice_cap_tw_modelled.k_racmo_upbound_volume_bsl.sum()


# Building the data arrays and total volume data frame
# FIRST FOR ALL GLACIERS
Areas = [df_vol_rest.Area.sum() + consensus_ice_cap.Area.sum(),
         df_vol_rest.Area.sum() + consensus_ice_cap.Area.sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum(),
         df_vol_rest.rgi_area_km2.sum() + df_vol_ice_cap['rgi_area_km2'].sum()]

Glaciers_total_volume = [df_vol_rest.vol_itmix_km3.sum(),
                         df_vol_rest.Huss_vol_km3.sum(),
                         df_vol_rest.inv_volume_km3.sum(),
                         df_vol_rest.k_measures_lowbound_inv_volume_km3.sum(),
                         df_vol_rest.k_measures_value_inv_volume_km3.sum(),
                         df_vol_rest.k_measures_upbound_inv_volume_km3.sum(),
                         df_vol_rest.k_itslive_lowbound_inv_volume_km3.sum(),
                         df_vol_rest.k_itslive_value_inv_volume_km3.sum(),
                         df_vol_rest.k_itslive_upbound_inv_volume_km3.sum(),
                         df_vol_rest.k_racmo_lowbound_inv_volume_km3.sum(),
                         df_vol_rest.k_racmo_value_inv_volume_km3.sum(),
                         df_vol_rest.k_racmo_upbound_inv_volume_km3.sum()]

Glaciers_total_volume_bsl = [df_vol_rest.vol_bsl_itmix_km3.sum(),
                             df_vol_rest.Huss_vol_bsl_km3.sum(),
                             df_vol_rest.volume_bsl.sum(),
                             df_vol_rest.k_measures_lowbound_volume_bsl.sum(),
                             df_vol_rest.k_measures_value_volume_bsl.sum(),
                             df_vol_rest.k_measures_upbound_volume_bsl.sum(),
                             df_vol_rest.k_itslive_lowbound_volume_bsl.sum(),
                             df_vol_rest.k_itslive_value_volume_bsl.sum(),
                             df_vol_rest.k_itslive_upbound_volume_bsl.sum(),
                             df_vol_rest.k_racmo_lowbound_volume_bsl.sum(),
                             df_vol_rest.k_racmo_value_volume_bsl.sum(),
                             df_vol_rest.k_racmo_upbound_volume_bsl.sum()]
#
Ice_cap_total_volume = [volume_ice_cap_Fari,
                        volume_ice_cap_Huss,
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

Ice_cap_total_volume_bsl = [volume_bsl_ice_cap_Fari,
                            volume_bsl_ice_cap_Huss,
                            oggm_ice_cap_no_calving_bsl,
                            k_measures_lowbound_ice_cap_vbsl,
                            k_measures_value_ice_cap_vbsl,
                            k_measures_upbound_ice_cap_vbsl,
                            k_itslive_lowbound_ice_cap_vbsl,
                            k_itslive_value_ice_cap_vbsl,
                            k_itslive_upbound_ice_cap_vbsl,
                            k_racmo_lowbound_ice_cap_vbsl,
                            k_racmo_value_ice_cap_vbsl,
                            k_racmo_upbound_ice_cap_vbsl]

total_vol = np.array(Glaciers_total_volume) + np.array(Ice_cap_total_volume)
print('Total volume')
print(configurations_order)
print(total_vol)


total_vol_bsl = np.array(Glaciers_total_volume_bsl) + np.array(Ice_cap_total_volume_bsl)
print('Total volume bsl')
print(configurations_order)
print(total_vol_bsl)

tol_vol_SLE = []
for vol, vol_bsl in zip(total_vol, total_vol_bsl):
    tol_vol_SLE.append(np.round(misc.compute_slr(vol-vol_bsl), 2))

print('Total volume sle')
print(tol_vol_SLE)

# volume_percentage_diff = []
#
# vol_no_calving = total_vol[2]
#
# print('Volume percentage differences between no calving and configurations')
# percentage = []
# for volss in total_vol[3:12]:
#     percentage.append(misc.calculate_volume_percentage(vol_no_calving,volss))
# print(configurations_order[3:12])
# print(percentage)
#
# print(configurations_order[9:12])
# vol_racmo = total_vol[9:12]
# print(configurations_order[3:6])
# vol_measures = total_vol[3:6]
#
# percentage_racmo_m = []
# for vol_one, vol_two in zip(vol_racmo, vol_measures):
#     percentage_racmo_m.append(misc.calculate_volume_percentage(vol_one,
#                                                                vol_two))
# print(percentage_racmo_m)
#
# print(configurations_order[9:12])
# print(configurations_order[6:9])
# vol_itslive = total_vol[6:9]
#
# percentage_racmo_i = []
# for vol_one, vol_two in zip(vol_racmo, vol_itslive):
#     percentage_racmo_i.append(misc.calculate_volume_percentage(vol_one,
#                                                                vol_two))
# print(percentage_racmo_i)
#
# print('Volume percentage differences between consensus and configurations')
# consensus = total_vol[0]
# huss = total_vol[1]
# percentage_con = []
# percentage_huss = []
# for volss in total_vol[3:12]:
#     percentage_con.append(misc.calculate_volume_percentage(consensus, volss))
#     percentage_huss.append(misc.calculate_volume_percentage(huss, volss))
# print(configurations_order[3:12])
# print(percentage_con)
# print(percentage_huss)
# print(np.mean(percentage_con[0:6]))
# print(np.mean(percentage_huss[0:6]))
#
# print('-------vol ---------')
# print(configurations_order[3:9])
# print('Mean and std volume for velocity methods',
#       np.round(np.mean(tol_vol_SLE[3:9]),2),
#       np.round(np.std(tol_vol_SLE[3:9]),2))
#
# print(configurations_order[9:12])
# print('Mean and std volume for racmo method',
#       np.round(np.mean(tol_vol_SLE[9:12]), 2),
#       np.round(np.std(tol_vol_SLE[9:12]), 2))
#
# print(configurations_order[3:9])
# print('Mean and std volume bsl for velocity methods',
#       np.round(np.mean(tol_vol_BSL_SLE[3:9]),2),
#       np.round(np.std(tol_vol_BSL_SLE[3:9]),2))
#
# print(configurations_order[9:12])
# print('Mean and std volume bsl for racmo method',
#       np.round(np.mean(tol_vol_BSL_SLE[9:12]), 2),
#       np.round(np.std(tol_vol_BSL_SLE[9:12]), 2))
#
# print(configurations_order[3:9])
# print('Mean and std volume - volume bsl for velocity methods',
#       np.round(np.mean(np.array(tol_vol_SLE[3:9])-np.array(tol_vol_BSL_SLE[3:9])),2),
#       np.round(np.std(np.array(tol_vol_SLE[3:9])-np.array(tol_vol_BSL_SLE[3:9])),2))
#
# print(configurations_order[9:12])
# print('Mean and std volume - volume bsl for racmo method',
#       np.round(np.mean(np.array(tol_vol_SLE[9:12])-np.array(tol_vol_BSL_SLE[9:12])), 2),
#       np.round(np.std(np.array(tol_vol_SLE[9:12])-np.array(tol_vol_BSL_SLE[9:12])), 2))
#
# value = np.round(np.mean(np.array(tol_vol_SLE[3:9])-np.array(tol_vol_BSL_SLE[3:9])),2)
# non_calving = tol_vol_SLE[2] - tol_vol_BSL_SLE[2]
# print('Percentage of change')
# print(misc.calculate_volume_percentage(non_calving, value))
#
#
#
df_volumes_total = {'Configuration': configurations_order,
                    'Area': Areas,
                    'Volume_all_glaciers': Glaciers_total_volume,
                    'Volume_ice_cap': Ice_cap_total_volume,
                    'Volume_all_glaciers_bsl': Glaciers_total_volume_bsl,
                    'Volume_ice_cap_bsl': Ice_cap_total_volume_bsl,
                    'Total_vol': total_vol,
                    'Total_vol_sle': tol_vol_SLE,
                    'Total_vol_bsl': total_vol_bsl}

data_frame = pd.DataFrame(data=df_volumes_total)

data_frame.to_csv(os.path.join(output_path +
                                   '/total_volume_vbsl_for_final_plot.csv'))