#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.
#set -e

export OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data_fix_tstar/01_Greenland_prepo/"
mkdir -p "$OUTDIR"

echo $OUTDIR

python ./run_calving_greenland_default.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -correct_width True -fix_t_star True

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

