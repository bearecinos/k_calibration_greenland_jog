# Calculate total glacier volume and frontal ablation flux per experiment
import os
import sys
import numpy as np
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

# Were to store merged data
output_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data')

# Calculate study area
study_area = 32202.540
print('Our target study area')
print(study_area)

dfa_rest = pd.read_csv(os.path.join(output_path,
                                    'fa_results_common_methods_rest.csv'))

dfa_ice_cap = pd.read_csv(os.path.join(output_path,
                                    'fa_results_common_methods_ice_cap.csv'))

sum_dfa = dfa_rest.iloc[:, 5:14].sum()

sum_dfa_ic = dfa_ice_cap.iloc[:, 5:14].sum()


total_fa = sum_dfa + sum_dfa_ic
print(total_fa)

#
total_fa_gt = total_fa/1.091
print(total_fa_gt)

print(total_fa_gt.index[1])

# Print estimates for the paper
print('----------- For de paper more information ------------------')
print('Mean and std Fa for velocity methods',
      np.round(np.mean(total_fa_gt[0:6]),2),
      np.round(np.std(total_fa_gt[0:6]),2))

print('Mean and std Fa for racmo method',
      np.round(np.mean(total_fa_gt[6:9]), 2),
      np.round(np.std(total_fa_gt[6:9]), 2))

print('Mean and std Fa for velocity methods',
      np.round(np.mean(total_fa[0:6]),2),
      np.round(np.std(total_fa[0:6]),2))

print('Mean and std Fa for racmo method',
      np.round(np.mean(total_fa[6:9]), 2),
      np.round(np.std(total_fa[6:9]), 2))
