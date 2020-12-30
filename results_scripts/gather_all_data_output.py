import os
import sys
import numpy as np
import glob
import pandas as pd
from configobj import ConfigObj

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import misc
from k_tools.utils_velocity import merge_vel_calibration_results_with_glac_stats
from k_tools.utils_racmo import merge_racmo_calibration_results_with_glac_stats

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Were to store merged data
output_path= os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test')
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Reading glacier directories per experiment
exp_dir_path = os.path.join(MAIN_PATH, config['volume_results'])

exp_dir_bvsl_path = os.path.join(MAIN_PATH, config['volume_bsl_results'])

config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

full_config_paths = []
for configuration in config_paths.config_path:
    full_config_paths.append(os.path.join(MAIN_PATH, exp_dir_path,
                                          configuration + '/'))

full_bvsl_config_paths = []
for configuration in config_paths.config_path:
    full_bvsl_config_paths.append(os.path.join(MAIN_PATH, exp_dir_bvsl_path,
                                               configuration + '/'))


# Merge velocity data with calibration results and volume below sea level
for path, path_vbsl, results in zip(full_config_paths[0:6],
                                    full_bvsl_config_paths[0:6],
                                    config_paths.fit_output[0:6]):

    experiment_name = misc.splitall(path)[-2]

    oggm_file_path = os.path.join(path,
                                  'glacier_statisticscalving_' +
                                  experiment_name +
                                  '.csv')

    oggm_vbsl_file_path = os.path.join(path_vbsl,
                                       experiment_name+
                                       '_volume_below_sea_level.csv')

    calibration_path = os.path.join(MAIN_PATH, config['linear_fit_to_data'],
                                    results + '.csv')

    df_merge = merge_vel_calibration_results_with_glac_stats(calibration_path,
                                                             oggm_file_path,
                                                             oggm_vbsl_file_path,
                                                             experiment_name)

    df_merge.to_csv(os.path.join(output_path,
                                 experiment_name+'_merge_results.csv'))


# Merge racmo data with calibration results
for path, path_vbsl, results in zip(full_config_paths[-3:],
                                    full_bvsl_config_paths[-3:],
                                    config_paths.fit_output[-3:]):

    experiment_name = misc.splitall(path)[-2]

    oggm_file_path = os.path.join(path,
                                  'glacier_statisticscalving_' +
                                  experiment_name +
                                  '.csv')

    oggm_vbsl_file_path = os.path.join(path_vbsl,
                                       experiment_name +
                                       '_volume_below_sea_level.csv')

    calibration_path = os.path.join(MAIN_PATH, config['linear_fit_to_data'],
                                    results + '.csv')

    df_merge = merge_racmo_calibration_results_with_glac_stats(calibration_path,
                                                               oggm_file_path,
                                                               oggm_vbsl_file_path,
                                                               experiment_name)
    df_merge.to_csv(os.path.join(output_path,
                                 experiment_name + '_merge_results.csv'))
