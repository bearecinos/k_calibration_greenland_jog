#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.
#set -e

export OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/02_Ice_cap_prepo/"
mkdir -p "$OUTDIR"

echo $OUTDIR

python ./run_calving_ice_cap_default.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

