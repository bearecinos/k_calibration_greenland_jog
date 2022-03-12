# Finds a range of k values per glacier that produce
# model velocities within the MEaSUREs velocity data observational range.
import os
import sys
import numpy as np
import pandas as pd
from configobj import ConfigObj
import glob
import pickle
from collections import defaultdict
import geopandas as gpd
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

# velocity module
from k_tools import utils_velocity as utils_vel

# Read output from k experiment
WORKING_DIR = os.path.join(MAIN_PATH, config['sensitivity_exp'], '*.csv')


# Read velocity observations from ITSlive
d_obs_itslive = pd.read_csv(os.path.join(MAIN_PATH,
                                         config['processed_vel_itslive']))

# Path to write out the data:
output_path = os.path.join(MAIN_PATH,
                           config['vel_calibration_results_itslive'])

# Read the RGI to store Area for statistics
rgidf = gpd.read_file(os.path.join(input_data_path, config['RGI_FILE']))
rgidf = rgidf.sort_values('RGIId', ascending=True)

# Read Areas for the ice-cap computed in OGGM during
# the pre-processing runs
df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))
df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

# Assign an area to the ice cap from OGGM to avoid errors
rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
          'Area'] = df_prepro_ic.rgi_area_km2.values

# Exclude glaciers with prepro-erros
#de = pd.read_csv(os.path.join(input_data_path, config['prepro_err']))
#ids = de.RGIId.values
#keep_errors = [(i not in ids) for i in rgidf.RGIId]
#rgidf = rgidf.iloc[keep_errors]

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Sort files
filenames = sorted(glob.glob(WORKING_DIR))

files_no_calving = []
files_no_vel_data = []
ids = []
Area_one = []
Area_two = []

output = defaultdict(list)

for j, f in enumerate(filenames):
    glacier = pd.read_csv(f)
    glacier = glacier.drop_duplicates(subset=('calving_flux'), keep=False)
    base = os.path.basename(f)
    rgi_id = os.path.splitext(base)[0]
    #print(rgi_id)
    #print(glacier)
    if glacier.empty:
        area = rgidf.Area.loc[rgidf.RGIId == rgi_id].values
        files_no_calving = np.append(files_no_calving, rgi_id)
        Area_one = np.append(Area_one, area)
    else:
        # Get observations for that glacier
        index_itslive = d_obs_itslive.index[d_obs_itslive['RGI_ID'] == rgi_id].tolist()

        if len(index_itslive) == 0:
            print('There is no Velocity data for this glacier' + rgi_id)
            files_no_vel_data = np.append(files_no_vel_data, rgi_id)
            area = rgidf.Area.loc[rgidf.RGIId == rgi_id].values
            Area_two = np.append(Area_two, area)
            continue
        else:
            # Perform the first step calibration and save the output as a
            # pickle file per glacier
            data_obs = d_obs_itslive.iloc[index_itslive]

            output[rgi_id] = utils_vel.find_k_values_within_vel_range(glacier,
                                                                      data_obs)
            fp = os.path.join(output_path, rgi_id + '.pkl')
            with open(fp, 'wb') as f:
                pickle.dump(output[rgi_id], f, protocol=-1)

# print(len(files_no_calving))
# print(len(Area_one))

d = {'RGIId': files_no_calving,
     'Area': Area_one}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(output_path,
                       'glaciers_with_no_solution.csv'))

s = {'RGIId': files_no_vel_data,
     'Area': Area_two}
ds = pd.DataFrame(data=s)
ds.to_csv(os.path.join(output_path,
                       'glaciers_with_no_vel_data.csv'))
