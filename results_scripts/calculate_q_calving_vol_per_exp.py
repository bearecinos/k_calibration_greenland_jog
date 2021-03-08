# Calculate total glacier volume and frontal ablation flux per experiment
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import salem
from configobj import ConfigObj

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
sys.path.append(MAIN_PATH)

from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Were to store merged data
output_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test')

# Reading glacier directories per experiment
configurations_order = ['k_itslive_lowbound',
                        'k_itslive_value',
                        'k_itslive_upbound',
                        'k_measures_lowbound',
                        'k_measures_value',
                        'k_measures_upbound',
                        'k_racmo_lowbound',
                        'k_racmo_value',
                        'k_racmo_upbound']

print(configurations_order)

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# Get glaciers that belong to the ice cap.
rgidf_ice_cap = rgidf[rgidf['RGIId'].str.match('RGI60-05.10315')]
# Get the id's for filter
ice_cap_ids = rgidf_ice_cap.RGIId.values

study_area = misc.get_study_area(rgidf, MAIN_PATH, config['ice_cap_prepro'])
print(study_area)

df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))

area_ice_cap_orig = df_prepro_ic.rgi_area_km2.sum()
print(area_ice_cap_orig)

df_prepro_ic_vbsl = pd.read_csv(os.path.join(MAIN_PATH,
                                             'output_data/02_Ice_cap_prepo/volume_below_sea_level.csv'),
                                index_col='Unnamed: 0')

df_prepro_ic_vbsl.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

errors = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']),
                     index_col='Unnamed: 0')
errors.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

errors_ic = errors[errors['rgi_id'].str.match('RGI60-05.10315')].copy()
errors_ic_ids = errors_ic.rgi_id.values

print('ice cap errors', errors_ic_ids)



area_no_measures = 204.752
area_no_solution = 3748.752
area_no_racmo = 105.845
area_no_itslive = 4417.071

exp_list = []
area = []
no_glaciers = []
volume_before_calving = []
volume_after_calving = []
calving_flux = []
volume_bsl = []
volume_bsl_c = []

for exp_name in configurations_order:
    exp_df = pd.read_csv(os.path.join(output_path,
                                      exp_name + '_merge_results.csv'),
                         index_col='Unnamed: 0')

    exp_df = exp_df[~exp_df.rgi_id.str.contains('RGI60-05.10315')]

    sum_exp = misc.summarize_exp(exp_df)

    exp_list = np.append(exp_list, exp_name)
    area = np.append(area, sum_exp.rgi_area_km2.sum())
    volume_before_calving = np.append(volume_before_calving,
                                      sum_exp.volume_before_calving.sum())
    volume_after_calving = np.append(volume_after_calving,
                                     sum_exp.inv_volume_km3.sum())

    calving_flux = np.append(calving_flux, sum_exp.calving_flux.sum())

    volume_bsl = np.append(volume_bsl, sum_exp.vbsl.sum())
    volume_bsl_c = np.append(volume_bsl_c, sum_exp.vbsl_c.sum())
    no_glaciers = np.append(no_glaciers, len(sum_exp))


# SORTING THE ICE CAP
ice_cap_area = []
ice_cap_no_basins = []
ice_cap_volume_before_calving = []
ice_cap_volume_after_calving = []
ice_cap_calving_flux = []
ice_cap_volume_bsl = []
ice_cap_volume_bsl_c = []

for exp_name in configurations_order:
    exp_df = pd.read_csv(os.path.join(output_path,
                                      exp_name + '_merge_results.csv'),
                         index_col='Unnamed: 0')

    exp_df = exp_df[exp_df['rgi_id'].str.match('RGI60-05.10315')].copy()


    df = exp_df[['rgi_id',
                 'rgi_area_km2',
                 'terminus_type',
                 'volume_before_calving',
                 'inv_volume_km3',
                 'calving_flux',
                 'vbsl',
                 'vbsl_c']]

    df['volume_before_calving'] = df.volume_before_calving * 1e-9

    # IDS to remove from prepro
    ids_to_remove = exp_df.rgi_id.values

    ids_out = np.append(errors_ic_ids, ids_to_remove)

    keep_ids = [(i not in ids_out) for i in df_prepro_ic.rgi_id]
    kee_ids_vbs = [(i not in ids_out) for i in df_prepro_ic_vbsl.rgi_id]

    df_prepro_ic = df_prepro_ic.loc[keep_ids]
    df_prepro_ic_vbsl = df_prepro_ic_vbsl.loc[kee_ids_vbs]

    df_prepro_ic_vbsl.rename(columns={'volume bsl': 'vbsl'}, inplace=True)

    df_ice_cap = df_prepro_ic[['rgi_id',
                               'rgi_area_km2',
                               'inv_volume_km3']]

    df_ice_cap.loc[:, 'terminus_type'] = df_prepro_ic_vbsl['terminus_type'].copy()

    df_ice_cap.loc[:, 'vbsl'] = df_prepro_ic_vbsl['vbsl'].copy()

    df_ice_cap.loc[:, 'volume_before_calving'] = df_ice_cap.loc[:,
                                                 'inv_volume_km3'].copy()
    df_ice_cap.loc[:, 'calving_flux'] = 0
    df_ice_cap.loc[:, 'vbsl_c'] = 0

    df_ice_cap = df_ice_cap[['rgi_id',
                             'rgi_area_km2',
                             'terminus_type',
                             'volume_before_calving',
                             'inv_volume_km3',
                             'calving_flux',
                             'vbsl',
                             'vbsl_c']]

    # print('Area before exp')
    # print(df_ice_cap.rgi_area_km2.sum())
    #
    # print('Area of exp')
    # print(df.rgi_area_km2.sum())

    df_final = df_ice_cap.append(df)
    # print('Area after exp')
    # print(df_final.rgi_area_km2.sum())


    ice_cap_area = np.append(ice_cap_area, df_final.rgi_area_km2.sum())

    ice_cap_volume_before_calving = np.append(ice_cap_volume_before_calving,
                                df_final.volume_before_calving.sum())

    ice_cap_volume_after_calving = np.append(ice_cap_volume_after_calving,
                                     df_final.inv_volume_km3.sum())

    ice_cap_calving_flux = np.append(ice_cap_calving_flux,
                                     df_final.calving_flux.sum())

    ice_cap_volume_bsl = np.append(ice_cap_volume_bsl,
                                   df_final.vbsl.sum())

    ice_cap_volume_bsl_c = np.append(ice_cap_volume_bsl_c,
                                     df_final.vbsl_c.sum())

    ice_cap_no_basins = np.append(ice_cap_no_basins, len(df_final))
#
#
total_fa = calving_flux + ice_cap_calving_flux

total_fa_gt = total_fa/1.091

print('icap area per exp')
ice_cap_area

df_calving_total = {'Configuration': exp_list,
                    'Area': area,
                    'No Glaciers': no_glaciers,
                    'Ice cap area': ice_cap_area,
                    'No Basins': ice_cap_no_basins,
                    'Volume before calving': volume_before_calving,
                    'Volume after calving': volume_after_calving,
                    'Volume before calving ice cap': ice_cap_volume_before_calving,
                    'Volume after calving ice cap': ice_cap_volume_after_calving,
                    'Calving flux': calving_flux,
                    'Calving flux ice cap': ice_cap_calving_flux,
                    'Volume bsl before calving': volume_bsl,
                    'Volume bsl after calving': volume_bsl_c,
                    'Volume bsl before calving ice cap': ice_cap_volume_bsl,
                    'Volume bsl after calving ice cap': ice_cap_volume_bsl_c,
                    'Total calving flux': total_fa,
                    'Total calving flux gt/yr': total_fa_gt}

data_frame = pd.DataFrame(data=df_calving_total)

data_frame.to_csv(os.path.join(output_path +
                               '/total_frontal_ablation_per_method.csv'))

# Print estimates for the paper
print('----------- For de paper more information ------------------')
print(exp_list[0:6])
print('Mean and std Fa for velocity methods',
      np.round(np.mean(total_fa_gt[0:6]),2),
      np.round(np.std(total_fa_gt[0:6]),2))

print(exp_list[6:9])
print('Mean and std Fa for racmo method',
      np.round(np.mean(total_fa_gt[6:9]), 2),
      np.round(np.std(total_fa_gt[6:9]), 2))

print(exp_list[0:6])
print('Mean and std Fa for velocity methods',
      np.round(np.mean(total_fa[0:6]),2),
      np.round(np.std(total_fa[0:6]),2))

print(exp_list[6:9])
print('Mean and std Fa for racmo method',
      np.round(np.mean(total_fa[6:9]), 2),
      np.round(np.std(total_fa[6:9]), 2))

print('----- No of glaciers  ------')
print(exp_list[1])
print(no_glaciers[1]+ice_cap_no_basins[1])
print(exp_list[3])
print(no_glaciers[3]+ice_cap_no_basins[3])
print(exp_list[6])
print(no_glaciers[6]+ice_cap_no_basins[6])


print('----- area coverage per data set ------')
print(exp_list[1])
print(area[1]+ice_cap_area[1]-area_no_itslive)
print(exp_list[3])
print(area[3]+ice_cap_area[3]-area_no_measures)
print(exp_list[6])
print(area[6]+ice_cap_area[6]-area_no_racmo)
print('----- % study area coverage per data set ------')
print(exp_list[1])
print((area[1]+ice_cap_area[1]-area_no_itslive-area_no_solution)*100/study_area)
print(exp_list[3])
print((area[3]+ice_cap_area[3]-area_no_measures-area_no_solution)*100/study_area)
print(exp_list[6])
print((area[6]+ice_cap_area[6]-area_no_racmo-area_no_solution)*100/study_area)