#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e

OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/01_Greenland_prepo/"
mkdir -p "$OUTDIR"

conda activate oggm_env
python ./run_calving_greenland_default.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini"

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

