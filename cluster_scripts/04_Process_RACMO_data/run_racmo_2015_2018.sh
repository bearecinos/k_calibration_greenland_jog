#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.

export OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/04_Process_RACMO_data/2015_2018/"
mkdir -p "$OUTDIR"

echo $OUTDIR

python ./get_racmo_smb_2015_2018.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

