# Calculate total glacier volume and frontal ablation flux per experiment
import os
import sys
import numpy as np
import glob
import pandas as pd
from configobj import ConfigObj
import argparse

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']
sys.path.append(MAIN_PATH)


# Results from all experiments
input_data = os.path.join(MAIN_PATH, config['volume_results'])

config_paths = pd.read_csv(os.path.join(config['input_data_folder'],
                                        config['configuration_names']))

output_path= os.path.join(MAIN_PATH, 'output_data/11_runs_stats')
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Read in Ice cap preprocessing
df_prepro_ic = pd.read_csv(os.path.join(config['input_data_folder'],
                                        config['ice_cap_prepro']))
print('Total ice cap basins')
print(len(df_prepro_ic))

# Exclude glaciers with prepro-erros from df_prepro_ic
de = pd.read_csv(os.path.join(config['input_data_folder'], config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in df_prepro_ic.rgi_id]
df_prepro_ic = df_prepro_ic.iloc[keep_errors]


exp_name = []
all_files = []
for exp in config_paths.config_path:
    name = exp.split("/")[-1]
    exp_path = os.path.join(input_data, exp)
    exp_name.append(name)
    all_files.append(glob.glob(exp_path + "/glacier_*.csv")[0])

# We read only one experiment to get the area
print(all_files[0])
print(all_files[3])
print(all_files[6])

df_itslive = pd.read_csv(all_files[0])
df_itslive_rest = df_itslive[~df_itslive.rgi_id.str.contains('RGI60-05.10315')].copy()
df_itslive_ice_cap = df_itslive[df_itslive['rgi_id'].str.match('RGI60-05.10315')].copy()

ic_ids_itslive = df_itslive_ice_cap.rgi_id.values

df_measures = pd.read_csv(all_files[3])
df_measures_rest = df_measures[~df_measures.rgi_id.str.contains('RGI60-05.10315')].copy()
df_measures_ice_cap = df_measures[df_measures['rgi_id'].str.match('RGI60-05.10315')].copy()

ic_ids_measures = df_measures_ice_cap.rgi_id.values

df_racmo = pd.read_csv(all_files[6])
df_racmo_rest = df_racmo[~df_racmo.rgi_id.str.contains('RGI60-05.10315')].copy()
df_racmo_ice_cap = df_racmo[df_racmo['rgi_id'].str.match('RGI60-05.10315')].copy()

ic_ids_racmo = df_racmo_ice_cap.rgi_id.values

# Remove from the preprocessing file the ice cap that
# we do model so we dont count that area
keep_no_model_itslive = [(i not in ic_ids_itslive) for i in df_prepro_ic.rgi_id]
keep_no_model_measures = [(i not in ic_ids_measures) for i in df_prepro_ic.rgi_id]
keep_no_model_racmo = [(i not in ic_ids_racmo) for i in df_prepro_ic.rgi_id]


df_prepro_ic_itslive = df_prepro_ic.iloc[keep_no_model_itslive]
df_prepro_ic_measures = df_prepro_ic.iloc[keep_no_model_measures]
df_prepro_ic_racmo = df_prepro_ic.iloc[keep_no_model_racmo]

# Read RACMO output for RACMO SMB value and find how many area has negative
# SMB
df_racmo_no_negative = df_racmo.drop(df_racmo[df_racmo.fa_racmo > 0].index)


# Calculate study area
study_area = 32202.540
print('Our target study area')
print(study_area)

print('Area coverage by ITSLIVE')
area_cover_itslive = df_itslive_rest.rgi_area_km2.sum() + \
                     df_prepro_ic_itslive.rgi_area_km2.sum() + \
                     df_itslive_ice_cap.rgi_area_km2.sum()
print(area_cover_itslive)
print('Percentage of study area')
area_itslive = (area_cover_itslive*100) / study_area
print(area_itslive)

print('Area coverage by MEASURES')
area_cover_measures = df_measures_rest.rgi_area_km2.sum() + \
                      df_prepro_ic_measures.rgi_area_km2.sum() + \
                      df_measures_ice_cap.rgi_area_km2.sum()
print(area_cover_measures)
print('Percentage of study area')
area_measures = (area_cover_measures*100) / study_area
print(area_measures)

print('Area coverage by RACMO')
area_cover_racmo = df_racmo_rest.rgi_area_km2.sum() + \
                   df_prepro_ic_racmo.rgi_area_km2.sum() + \
                   df_racmo_ice_cap.rgi_area_km2.sum()
print(area_cover_racmo)
print('Percentage of study area')
area_racmo = (area_cover_racmo*100) / study_area
print(area_racmo)

print('Percentage of study area with a negative SMB')
area_racmo_neg = (df_racmo_no_negative.rgi_area_km2.sum()*100) / study_area
print(area_racmo_neg)

print(area_racmo-area_racmo_neg)

category = ['Area modeled with ITSLIVE',
            'Area modeled by MEaSUREs',
            'Area modeled by RACMO',
            'Area where RACMO has negative SMB']

area = [area_cover_itslive,
        area_cover_measures,
        area_cover_racmo,
        area_racmo_neg]

area_percent = area/study_area*100

d = {'Category': category,
     'Area (km²)': area,
     'Area (% of Greenland)': area_percent}
ds = pd.DataFrame(data=d)

ds.to_csv(os.path.join(output_path, 'area_stats.csv'))