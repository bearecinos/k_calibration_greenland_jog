#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.

export OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/05_k_exp_for_calibration/"
mkdir -p "$OUTDIR"

echo $OUTDIR

python ./k_parameter_exp.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -correct_width True -fix_t_star True

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

