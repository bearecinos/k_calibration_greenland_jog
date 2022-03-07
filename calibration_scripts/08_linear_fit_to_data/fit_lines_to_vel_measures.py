# Finds 3 k values per glacier that produce
# model surface velocities, within
# MEaSUREs data range.
# K values are found by finding the intercepts between linear equations
# fitted to model and observations values.
import os
import sys
import numpy as np
from configobj import ConfigObj
import glob
import pickle
from scipy.stats import linregress
import pandas as pd
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

# Misc module
from k_tools import misc as misc

# path to save data to
output_path = os.path.join(MAIN_PATH, config['linear_fit_to_data'])
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Sort files
filenames = sorted(glob.glob(os.path.join(MAIN_PATH,
                                          config['vel_calibration_results_measures'],
                                          "*.pkl")))

# Long list of variables to save after calibration.
ids = []
messages = []
vel_obs = []
vel_lwl = []
vel_upl = []
k_v = []
k_lw = []
k_up = []
model_slope = []
model_intercept = []
r_values = []
p_values = []
std_errs = []

for file in filenames:
    with open(file, 'rb') as handle:
        base = os.path.basename(file)
        rgi_id = os.path.splitext(base)[0]
        print(rgi_id)
        g = pickle.load(handle)

        # Observations slope, intercept. y = ax + b
        # where a is zero and b is the observation
        obs = g['obs_vel'][0].vel_calving_front.iloc[0]
        low_bound = g['low_lim_vel'][0][0]
        up_bound = g['up_lim_vel'][0][0]

        slope_obs, intercept_obs = [0.0, obs]
        slope_lwl, intercept_lwl = [0.0, low_bound]
        slope_upl, intercept_upl = [0.0, up_bound]

        # Get linear fit for OGGM model data

        # Get the model data from the first calibration step
        df_oggm = g['oggm_vel'][0]
        warning = g['vel_message'][0]
        # If there is only one model value (k1, vel1) e.g. when oggm
        # overestimates or underestimates velocities we then add (0,0)
        # as another point along the line. k=0 and velocity=0 is a
        # valid solution
        if len(df_oggm.index.values) <= 2:
            df_oggm.loc[len(df_oggm) + 1] = 0.0
            df_oggm = df_oggm.sort_values(by=['k_values']).reset_index(drop=True)
        else:
            df_oggm = df_oggm

        k_values = df_oggm.k_values.values
        velocities = df_oggm.velocity_surf.values

        # Get the equation for the model data. y = ax + b
        slope, intercept, r_value, p_value, std_err = linregress(k_values,
                                                                 velocities)

        Z_value = misc.solve_linear_equation(slope_obs, intercept_obs,
                                             slope, intercept)

        Z_lower_bound = misc.solve_linear_equation(slope_lwl, intercept_lwl,
                                             slope, intercept)

        Z_upper_bound = misc.solve_linear_equation(slope_upl, intercept_upl,
                                             slope, intercept)

        # Saving the intercept to observations and linear fit statistics
        ids = np.append(ids, rgi_id)
        messages = np.append(messages, warning)
        vel_obs = np.append(vel_obs, obs)
        vel_lwl = np.append(vel_lwl, low_bound)
        vel_upl = np.append(vel_upl, up_bound)
        k_v = np.append(k_v, Z_value[0])
        np.clip(k_v, 0, None, out=k_v)
        k_lw = np.append(k_lw, Z_lower_bound[0])
        np.clip(k_lw, 0, None, out=k_lw)
        k_up = np.append(k_up, Z_upper_bound[0])
        np.clip(k_up, 0, None, out=k_up)
        model_slope = np.append(model_slope, slope)
        model_intercept = np.append(model_intercept, intercept)
        r_values = np.append(r_values, r_value)
        p_values = np.append(p_values, p_value)
        std_errs = np.append(std_errs, std_err)


dk = {'RGIId': ids,
      'method': messages,
      'surface_vel_obs': vel_obs,
      'obs_low_bound': vel_lwl,
      'obs_up_bound': vel_upl,
      'k_for_obs_value': k_v,
      'k_for_lw_bound': k_lw,
      'k_for_up_bound': k_up,
      'model_fit_slope': model_slope,
      'model_fit_intercept': model_intercept,
      'model_fit_r_value': r_value,
      'model_fit_p_value': p_value,
      'model_fit_std_error': std_err}

df = pd.DataFrame(data=dk)
df.to_csv(os.path.join(output_path, 'velocity_fit_calibration_results_measures.csv'))


