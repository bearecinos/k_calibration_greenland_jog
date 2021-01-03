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

# Where we stored merged data
output_path= os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test')

output_path_vel = os.path.join(MAIN_PATH,
                               'output_data/12_Calculate_vel_with_k_results')

config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))
exp_name = []
for exp in config_paths.config_path:
    name = exp.split("/")[-1]
    exp_name.append(name)

# Find common glaciers among all data sets and get the data that will be the
# same for all k values.
all_files = sorted(glob.glob(output_path + "/*.csv"))

files_vel = []
for exp, subpath in zip(exp_name, config_paths.config_path):
    path = os.path.join(subpath, exp)
    file_name = os.path.join(output_path_vel, path + '_velocity_data.csv')
    files_vel.append(file_name)

li = []
for filename in all_files[1:-2]:
    df = pd.read_csv(filename, index_col=0)
    li.append(df)

core = []
for d in li:
    core_data = misc.get_core_data(d)
    core.append(core_data)

dep = []
for d, name, vel_file in zip(li, exp_name, files_vel):
    df_vel = pd.read_csv(vel_file)
    dep_data = misc.get_k_dependent(d, df_vel, name)
    dep.append(dep_data)

df_common = reduce(lambda x,y: pd.merge(x,y, on='rgi_id', how='inner'), dep)


df_final = pd.concat([core[0], df_common], join='inner', axis=1)

df_final.to_csv(os.path.join(output_path, 'common_final_results.csv'))




