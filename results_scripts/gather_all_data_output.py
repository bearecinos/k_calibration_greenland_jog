import os
import sys
import glob
import pandas as pd
from configobj import ConfigObj
from functools import reduce

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_jog/')
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

print(all_files[0:6])

its_live_li = []
for filename, name in zip(all_files[0:3], exp_name[0:3]):
    df = pd.read_csv(filename)
    its_live_li.append(df)

# Now lets find the glaciers for which we have data among all methods
df_itslive = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'),
                        its_live_li)

df_itslive.to_csv(os.path.join(output_path,
                           'glaciers_itslive.csv'))

measures_li = []
for filename, name in zip(all_files[3:6], exp_name[3:6]):
    df = pd.read_csv(filename)
    measures_li.append(df)

# Now lets find the glaciers for which we have data among all methods
df_measures = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'),
                        measures_li)
df_measures.to_csv(os.path.join(output_path,
                           'glaciers_measures.csv'))

print(all_files[0:3] + all_files[6:10])

its_live_racmo_li = []
for filename, name in zip(all_files[0:3] + all_files[6:10],
                          exp_name[0:3] + exp_name[6:10]):
    df = pd.read_csv(filename)
    its_live_racmo_li.append(df)

# Now lets find the glaciers for which we have data among all methods
df_itslive_racmo = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'),
                        its_live_racmo_li)

df_itslive_racmo.to_csv(os.path.join(output_path,
                           'common_glaciers_itslive_racmo.csv'))


print(all_files[3:6] + all_files[6:10])

measures_racmo_li = []
for filename, name in zip(all_files[3:6] + all_files[6:10],
                          exp_name[3:6] + exp_name[6:10]):
    df = pd.read_csv(filename)
    measures_racmo_li.append(df)

# Now lets find the glaciers for which we have data among all methods
df_measures_racmo = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'),
                        measures_racmo_li)

df_measures_racmo.to_csv(os.path.join(output_path,
                           'common_glaciers_measures_racmo.csv'))
