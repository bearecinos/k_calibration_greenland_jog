#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.

export OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/04_Process_RACMO_data/1961_1990/"
mkdir -p "$OUTDIR"

echo $OUTDIR

python ./get_racmo_smb31_1961_1990.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

