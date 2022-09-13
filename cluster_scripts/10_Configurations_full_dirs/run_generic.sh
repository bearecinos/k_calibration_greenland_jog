#!/bin/bash
export k_exp=k_racmo

echo $k_exp

export OUTDIR_low="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/10_Configurations_full_dirs/${k_exp}/${k_exp}_lowbound"
export OUTDIR_obs="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/10_Configurations_full_dirs/${k_exp}/${k_exp}_value"
export OUTDIR_up="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/10_Configurations_full_dirs/${k_exp}/${k_exp}_upbound"
mkdir -p "$OUTDIR_low"
mkdir -p "$OUTDIR_obs"
mkdir -p "$OUTDIR_up"

echo $OUTDIR_low
echo $OUTDIR_obs
echo $OUTDIR_up

python ${k_exp}/${k_exp}_lowbound.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -correct_width True
python ${k_exp}/${k_exp}_value.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -correct_width True
python ${k_exp}/${k_exp}_upbound.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -correct_width True

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"
