#!/bin/bash
for script in k_itslive/*; do sbatch ./run_generic_singularity.slurm "$script"; done
