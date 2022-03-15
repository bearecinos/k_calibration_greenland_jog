import pandas as pd
import os
import sys
import geopandas as gpd
import numpy as np
import salem
from configobj import ConfigObj
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

# RGI file
rgidf = gpd.read_file(os.path.join(input_data_path, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# Run only for Lake Terminating and Marine Terminating
glac_type = ['0']
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Run only glaciers that have a week connection or are
# not connected to the ice-sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Exclude glaciers with prepro-erros
de = pd.read_csv(os.path.join(input_data_path, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

ids = rgidf.RGIId.values

# Reading thickness data form Millan et all 2022
path_h = os.path.join(MAIN_PATH, config['thickness_obs'])

for i in ids:
    file_paths = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_h):
        for file in f:
            if i + '.csv' == file:
                file_paths.append(os.path.join(r, file))

    if  len(file_paths) > 0 and os.path.isfile(file_paths[0]):
        df = pd.DataFrame()
        for item in file_paths:
            data_f = pd.read_csv(item, index_col=0)
            shape_ori = data_f.shape
            df = pd.concat([df, data_f], axis=1).reset_index(drop=True)

        df_drop = df.loc[:, ~df.T.duplicated(keep='first')]

        if len(df_drop.columns) == 4:
            assert df_drop.shape == shape_ori # sanity check
            df_drop.to_csv(os.path.join(path_h, i + '.csv'))
        elif len(df_drop.columns) > 4:
            df_drop.to_csv(os.path.join(path_h + '/to_check', i + '_tocheck.csv'))
        else:
            print(len(df_drop.columns))
            print(i)
            if not (df_drop['H_flowline'] == 0).sum() == len(df_drop):
                df_drop.to_csv(os.path.join(path_h + '/to_check', i + '_tocheck.csv'))
            else:
                print('There is no data for this glacier' + i)
