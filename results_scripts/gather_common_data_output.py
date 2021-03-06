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

# output_path_vel = os.path.join(MAIN_PATH,
#                                'output_data/12_Calculate_vel_with_k_results')

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

df_common = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'), li)

df_common.to_csv(os.path.join(output_path, 'common_final_results.csv'))




