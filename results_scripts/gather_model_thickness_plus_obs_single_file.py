import pandas as pd
import os
import sys
import numpy as np
from configobj import ConfigObj
from collections import defaultdict
import argparse

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
args = parser.parse_args()
config_file = args.conf

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']
input_data_path = config['input_data_folder']
sys.path.append(MAIN_PATH)

from k_tools import utils_thick

marcos_data = os.path.join(MAIN_PATH, 'output_data_marco')

config_paths = pd.read_csv(os.path.join(input_data_path,
                                        config['configuration_names']))

full_config_paths = []
for confs in config_paths.config_path:
    full_config_paths.append(os.path.join(MAIN_PATH, marcos_data,
                                          confs.split("/")[-1]))

print(full_config_paths[6:9])
# Create an empty dictionary to drop all data per config
d = defaultdict(list)

# This loop should be done by main experiment!

for path_config in full_config_paths[6:9]:
    rgi_files = os.listdir(path_config)
    ids = []
    model = []
    obs = []
    errors = []

    for file in rgi_files:
        path_full = os.path.join(path_config, file)
        out = utils_thick.combined_model_thickness_and_observations(path_full)
        if not out:
            continue
        else:
            ids = np.append(ids, out[0])
            model = np.append(model, out[1])
            obs = np.append(obs, out[2])
            errors = np.append(errors, out[3])
    name = path_config.split("/")[-1]
    df = {'rgi_id': ids,
          'H_model_'+name: model,
          'H_obs': obs,
          'H_error': errors
          }
    ds = pd.DataFrame(data=df)
    d[name] = ds

keys = list(d.keys())

df_final = pd.DataFrame()

for key in keys:
    data = d[key]
    df_final = pd.concat([df_final, data], axis=1).reset_index(drop=True)

df_drop = df_final.loc[:, ~df_final.T.duplicated(keep='first')]
df_drop.to_csv(os.path.join(marcos_data, 'racmo' + '.csv'))