#!/bin/bash
for script in k_RACMO/*; do sbatch ./run_generic_singularity.slurm "$script"; done
