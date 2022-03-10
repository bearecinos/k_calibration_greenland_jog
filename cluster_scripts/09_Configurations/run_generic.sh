#!/bin/bash

export OUTDIR_low="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/09_Configurations/${1}/${1}_lowbound"
export OUTDIR_obs="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/09_Configurations/${1}/${1}_value"
export OUTDIR_up="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/09_Configurations/${1}/${1}_upbound"
mkdir -p "$OUTDIR_low"
mkdir -p "$OUTDIR_obs"
mkdir -p "$OUTDIR_up"

echo $OUTDIR_low
echo $OUTDIR_obs
echo $OUTDIR_up

#python ${1}/${1}_lowbound.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"
#python ${1}/${1}_value.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"
#python ${1}/${1}_upbound.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"
