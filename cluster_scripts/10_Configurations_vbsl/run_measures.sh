#!/bin/bash
for script in k_measures/*; do sbatch ./run_generic_singularity.slurm "$script"; done
