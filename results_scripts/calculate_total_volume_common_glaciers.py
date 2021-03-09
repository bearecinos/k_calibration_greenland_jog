# Calculate total glacier volume for final plot
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import salem
from configobj import ConfigObj

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
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

# Calculate study area
study_area = 32202.540
print('Our target study area')
print(study_area)

# Read common glaciers data frame and select the data that we need
df_vol_rest = pd.read_csv(os.path.join(output_path,
                                       'volume_rest_all_methos_plus_consensus.csv'))
print('Number of glaciers modelled')
print(len(df_vol_rest))


# Read ice cap results
df_vol_ice_cap = pd.read_csv(os.path.join(output_path,
                                       'ice_cap_volume_all_methos_plus_consensus.csv'))

print('Number of ice cap basins')
print(len(df_vol_ice_cap))

# Building the data arrays and total volume data frame
# FIRST FOR ALL GLACIERS
Areas = [df_vol_rest.Area.sum() + df_vol_ice_cap.Area.sum(),
         df_vol_rest.Area.sum() + df_vol_ice_cap.Area.sum(),
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
Ice_cap_total_volume = [df_vol_ice_cap.vol_itmix_km3.sum(),
                        df_vol_ice_cap.Huss_vol_km3.sum(),
                        df_vol_ice_cap.inv_volume_km3.sum(),
                        df_vol_ice_cap.k_measures_lowbound_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_measures_value_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_measures_upbound_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_itslive_lowbound_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_itslive_value_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_itslive_upbound_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_racmo_lowbound_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_racmo_value_inv_volume_km3.sum(),
                        df_vol_ice_cap.k_racmo_upbound_inv_volume_km3.sum()]

Ice_cap_total_volume_bsl = [df_vol_ice_cap.vol_bsl_itmix_km3.sum(),
                            df_vol_ice_cap.Huss_vol_bsl_km3.sum(),
                            df_vol_ice_cap.volume_bsl.sum(),
                            df_vol_ice_cap.k_measures_lowbound_volume_bsl.sum(),
                            df_vol_ice_cap.k_measures_value_volume_bsl.sum(),
                            df_vol_ice_cap.k_measures_upbound_volume_bsl.sum(),
                            df_vol_ice_cap.k_itslive_lowbound_volume_bsl.sum(),
                            df_vol_ice_cap.k_itslive_value_volume_bsl.sum(),
                            df_vol_ice_cap.k_itslive_upbound_volume_bsl.sum(),
                            df_vol_ice_cap.k_racmo_lowbound_volume_bsl.sum(),
                            df_vol_ice_cap.k_racmo_value_volume_bsl.sum(),
                            df_vol_ice_cap.k_racmo_upbound_volume_bsl.sum()]


total_vol = np.array(Glaciers_total_volume) #+ np.array(Ice_cap_total_volume)
print('Total volume')
print(configurations_order)
print(total_vol)


total_vol_bsl = np.array(Glaciers_total_volume_bsl) #+ np.array(Ice_cap_total_volume_bsl)
print('Total volume bsl')
print(configurations_order)
print(total_vol_bsl)

tol_SL_contribution = []
for vol, vol_bsl in zip(total_vol, total_vol_bsl):
    tol_SL_contribution.append(np.round(misc.compute_slr(vol-vol_bsl), 2))
print('Sea level contribution per method')
print(tol_SL_contribution)

tol_vol_SLE = []
tol_vol_BSL_SLE = []

for vol, vol_bsl in zip(total_vol, total_vol_bsl):
    tol_vol_SLE.append(
        np.round(abs(misc.calculate_sea_level_equivalent(vol)),2))
    tol_vol_BSL_SLE.append(
        np.round(abs(misc.calculate_sea_level_equivalent(vol_bsl)), 2))

print('Total volume sle')
print(tol_vol_SLE)

volume_percentage_diff = []

vol_no_calving = total_vol[2]

print('Volume percentage differences between no calving and configurations')
percentage = []
for volss in total_vol[3:12]:
    percentage.append(misc.calculate_volume_percentage(vol_no_calving,volss))
print(configurations_order[3:12])
print(percentage)

print(configurations_order[9:12])
vol_racmo = total_vol[9:12]
print(configurations_order[3:6])
vol_measures = total_vol[3:6]
#
percentage_racmo_m = []
for vol_one, vol_two in zip(vol_racmo, vol_measures):
    percentage_racmo_m.append(misc.calculate_volume_percentage(vol_one,
                                                               vol_two))
print('Volume percentage differences between racmo and measures')
print(percentage_racmo_m)
#
print(configurations_order[9:12])
print(configurations_order[6:9])
vol_itslive = total_vol[6:9]

percentage_racmo_i = []
for vol_one, vol_two in zip(vol_racmo, vol_itslive):
    percentage_racmo_i.append(misc.calculate_volume_percentage(vol_one,
                                                               vol_two))
print('Volume percentage differences between racmo and itslive')
print(percentage_racmo_i)

print('Volume percentage differences between consensus and configurations')
consensus = total_vol[0]
huss = total_vol[1]
percentage_con = []
percentage_huss = []
for volss in total_vol[3:12]:
    percentage_con.append(misc.calculate_volume_percentage(consensus, volss))
    percentage_huss.append(misc.calculate_volume_percentage(huss, volss))
print(configurations_order[3:12])
print(percentage_con)
print(percentage_huss)
print(np.mean(percentage_con[0:6]))
print(np.mean(percentage_huss[0:6]))

print('-------vol ---------')
print(configurations_order[3:9])
print('Mean and std volume for velocity methods',
      np.round(np.mean(tol_SL_contribution[3:9]),2),
      np.round(np.std(tol_SL_contribution[3:9]),2))

print(configurations_order[9:12])
print('Mean and std volume for racmo method',
      np.round(np.mean(tol_SL_contribution[9:12]), 2),
      np.round(np.std(tol_SL_contribution[9:12]), 2))

print(configurations_order[3:9])
print('Mean and std volume bsl for velocity methods',
      np.round(np.mean(tol_vol_BSL_SLE[3:9]),2),
      np.round(np.std(tol_vol_BSL_SLE[3:9]),2))

print(configurations_order[9:12])
print('Mean and std volume bsl for racmo method',
      np.round(np.mean(tol_vol_BSL_SLE[9:12]), 2),
      np.round(np.std(tol_vol_BSL_SLE[9:12]), 2))

print(configurations_order[3:9])
print('Mean and std volume - volume bsl for velocity methods',
      np.round(np.mean(np.array(tol_vol_SLE[3:9])-np.array(tol_vol_BSL_SLE[3:9])),2),
      np.round(np.std(np.array(tol_vol_SLE[3:9])-np.array(tol_vol_BSL_SLE[3:9])),2))

print(configurations_order[9:12])
print('Mean and std volume - volume bsl for racmo method',
      np.round(np.mean(np.array(tol_vol_SLE[9:12])-np.array(tol_vol_BSL_SLE[9:12])), 2),
      np.round(np.std(np.array(tol_vol_SLE[9:12])-np.array(tol_vol_BSL_SLE[9:12])), 2))

value = np.round(np.mean(np.array(tol_vol_SLE[3:9])-np.array(tol_vol_BSL_SLE[3:9])),2)
non_calving = tol_vol_SLE[2] - tol_vol_BSL_SLE[2]
print('Percentage of change')
print(misc.calculate_volume_percentage(non_calving, value))

df_volumes_total = {'Configuration': configurations_order,
                    'Area': Areas,
                    'Volume_all_glaciers': Glaciers_total_volume,
                    'Volume_ice_cap': Ice_cap_total_volume,
                    'Volume_all_glaciers_bsl': Glaciers_total_volume_bsl,
                    'Volume_ice_cap_bsl': Ice_cap_total_volume_bsl,
                    'Total_vol': total_vol,
                    'Total_vol_sle': tol_vol_SLE,
                    'Total_vol_bsl': total_vol_bsl,
                    'Total_vol_bsl_sle': tol_vol_BSL_SLE}

data_frame = pd.DataFrame(data=df_volumes_total)

exit()
data_frame.to_csv(os.path.join(output_path +
                                   '/total_volume_vbsl_for_final_plot.csv'))