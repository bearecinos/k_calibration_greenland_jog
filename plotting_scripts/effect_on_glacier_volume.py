import os
import sys
import numpy as np
from configobj import ConfigObj
import geopandas as gpd
import salem
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import pandas as pd

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland_new/')
sys.path.append(MAIN_PATH)

from k_tools import utils_velocity as utils_vel
from k_tools import utils_racmo as utils_racmo
from k_tools import misc as misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/13_Merged_data_test/')

df = pd.read_csv(os.path.join(output_dir_path,
                                    'total_volume_vbsl_for_final_plot.csv'))

configuration = df['Configuration'].values
print(configuration)
ice_cap_vol_bsl_exp = df['Volume_ice_cap_bsl'].values
vol_bsl_exp = df['Volume_all_glaciers_bsl'].values
vol_exp = df['Volume_all_glaciers'].values

ice_cap_vol_exp = df['Volume_ice_cap'].values
print('volumes')
print(ice_cap_vol_exp+vol_exp)

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf.crs = salem.wgs84.srs

# Calculate study area
study_area = misc.get_study_area(rgidf, MAIN_PATH, config['ice_cap_prepro'])
print('Our target study area')
print(study_area)

print('Percentage of study area')
area = (df['Area']*100) / study_area
print(area)


fig = plt.figure(figsize=(14, 8))
sns.set(style="white", context="talk")

ax1=fig.add_subplot(111)
color_palette = sns.color_palette("deep")

# color_array = [color_palette[3], color_palette[4],
#                color_palette[2], color_palette[0],
#                color_palette[3], color_palette[4],
#                color_palette[2], color_palette[1],
color_array = [color_palette[3], color_palette[4], color_palette[5],
               color_palette[0], color_palette[0], color_palette[0],
               color_palette[2], color_palette[2], color_palette[2],
               color_palette[1], color_palette[1], color_palette[1]]

# Example data
y_pos = [0,0.5,1,2,2.5,3,4,4.5,5,6,6.5,7]


p0 = ax1.barh(y_pos, (ice_cap_vol_bsl_exp+vol_bsl_exp)*-1,
              align='center', color=sns.xkcd_rgb["grey"],
              height=0.5, edgecolor="white", hatch="/")

p1 = ax1.barh(y_pos, vol_bsl_exp*-1, align='center',
              color=sns.xkcd_rgb["grey"], height=0.5, edgecolor="white")

p2 = ax1.barh(y_pos, vol_exp+ice_cap_vol_exp, align='center',
              color=color_array, height=0.5, hatch="/")

p3 = ax1.barh(y_pos, vol_exp, align='center', color=color_array, height=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([])
# labels read top-to-bottom
ax1.invert_yaxis()
ax1.set_xlabel('Volume [km³]',fontsize=18)

ax1.set_xticks([-4000, -2000, 0, 2000, 4000, 6000])
ax1.set_xticklabels([4000, 2000, 0, 2000, 4000, 6000], fontsize=20)
ax2= ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
array = ax1.get_xticks()

#Get the other axis on sea level equivalent
sle = []
for value in array:
    sle.append(np.round(abs(misc.calculate_sea_level_equivalent(value)),2))
print(sle)

ax2.set_xticklabels(sle, fontsize=20)
ax2.set_xlabel('Volume [mm SLE]', fontsize=18)

plt.legend((p3[0], p3[1], p3[2], p3[3], p3[6], p3[9]),
           ('Farinotti et al. (2019)',
            'Huss and Farinotti. (2012)',
            'Without $q_{calving}$',
            'With $q_{calving}$ - MEaSUREs',
            'With $q_{calving}$ - ITSlive',
            'With $q_{calving}$ - RACMO'),
            frameon=True, bbox_to_anchor=(0.8, -0.2), ncol=2)
            #bbox_to_anchor=(1.1, -0.15), ncol=5, fontsize=15)

plt.margins(0.05)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plot_path, 'volume_greenland_with_ice_cap_common.pdf'),
                bbox_inches='tight')