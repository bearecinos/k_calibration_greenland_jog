# Finds 3 k values per glacier that produce
# model frontal ablation fluxes, within
# RACMO frontal ablation estimates data range.
# K values are found by finding the intercepts between linear equations
# fitted to model and RACMO values.
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
                                          config['racmo_calibration_results'],
                                          "*.pkl")))

# Long list of variables to save after calibration.
ids = []
messages = []
fa_racmo = []
fa_racmo_lwl = []
fa_racmo_upl = []
k_v = []
k_lw = []
k_up = []
model_slope = []
model_intercept = []
r_values = []
p_values = []
std_errs = []

ids_with_negative_flux = []

for file in filenames:
    with open(file, 'rb') as handle:
        base = os.path.basename(file)
        rgi_id = os.path.splitext(base)[0]
        print(rgi_id)
        g = pickle.load(handle)

        # Observations slope, intercept. y = ax + b
        # where a is zero and b is the observation
        obs = g['obs_racmo'][0].q_calving_RACMO_mean.iloc[0]
        low_bound = g['low_lim_racmo'][0][0]
        up_bound = g['up_lim_racmo'][0][0]

        # Make sure tha if uncertainty and RACMO value are always zero if
        # RACMO calving flux estimate is negative.
        # if so k should be zero.
        if obs < 0:
            obs = 0.0
        if low_bound < 0:
            low_bound = 0.0
        if up_bound < 0:
            up_bound = 0.0

        slope_obs, intercept_obs = [0.0, obs]
        slope_lwl, intercept_lwl = [0.0, low_bound]
        slope_upl, intercept_upl = [0.0, up_bound]

        # Get linear fit for OGGM model data
        # Get the model data from the first calibration step
        df_oggm = g['oggm_racmo'][0]
        warning = g['racmo_message'][0]

        if warning == 'This glacier should not calve':
            print(rgi_id)
            Z_value = [0.0, 0.0]
            Z_lower_bound = [0.0, 0.0]
            Z_upper_bound = [0.0, 0.0]
        else:
            # If there is only one model value (k1, Fa1) e.g. when oggm
            # overestimates Frontal ablation
            # we then add (0,0) as another point along the data model fit.
            # k=0 and Fa=0 is a valid solution.
            if len(df_oggm.index.values) <= 2:
                df_oggm.loc[len(df_oggm) + 1] = 0.0
                df_oggm = df_oggm.sort_values(by=['k_values']).reset_index(drop=True)
            else:
                df_oggm = df_oggm

            k_values = df_oggm.k_values.values
            calving_fluxes = df_oggm.calving_flux.values

            # Get the equation for the model data. y = ax + b
            slope, intercept, r_value, p_value, std_err = linregress(k_values,
                                                                     calving_fluxes)

            Z_value = misc.solve_linear_equation(slope_obs,
                                                 intercept_obs,
                                                 slope, intercept)

            Z_lower_bound = misc.solve_linear_equation(slope_lwl,
                                                       intercept_lwl,
                                                       slope, intercept)

            Z_upper_bound = misc.solve_linear_equation(slope_upl,
                                                       intercept_upl,
                                                       slope, intercept)

    # Saving the intercept to observations and linear fit statistics
    ids = np.append(ids, rgi_id)
    messages = np.append(messages, warning)
    fa_racmo = np.append(fa_racmo, g['obs_racmo'][0].q_calving_RACMO_mean.iloc[0])
    fa_racmo_lwl = np.append(fa_racmo_lwl, g['low_lim_racmo'][0][0])
    fa_racmo_upl = np.append(fa_racmo_upl, g['up_lim_racmo'][0][0])
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
      'fa_racmo': fa_racmo,
      'racmo_low_bound': fa_racmo_lwl,
      'racmo_up_bound': fa_racmo_lwl,
      'k_for_racmo_value': k_v,
      'k_for_lw_bound': k_lw,
      'k_for_up_bound': k_up,
      'model_fit_slope': model_slope,
      'model_fit_intercept': model_intercept,
      'model_fit_r_value': r_value,
      'model_fit_p_value': p_value,
      'model_fit_std_error': std_err}

df = pd.DataFrame(data=dk)
df.to_csv(os.path.join(output_path, 'racmo_fit_calibration_results.csv'))


