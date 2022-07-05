#!/bin/bash
# Abort whenever a single step fails. Without this, bash will just continue on errors.

export OUTDIR="/scratch/local/brecinos/k_calibration_greenland_jog/output_data/03_Process_velocity_data/thickness/"
mkdir -p "$OUTDIR"

echo $OUTDIR

python ./process_thickness_data_version2.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -raster_start 0 -raster_end 2
python ./process_thickness_data_version2.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -raster_start 2 -raster_end 2
python ./process_thickness_data_version2.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -raster_start 4 -raster_end 6
python ./process_thickness_data_version2.py -conf "/scratch/local/brecinos/k_calibration_greenland_jog/config.ini" -raster_start 6 -raster_end 7

# Print a final message so you can actually see it being done in the output log.
echo "RUN DONE"

