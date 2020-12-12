# OGGM k calibration

This repository contains the scripts used to calibrate a frontal ablation parameterization in [OGGM](https://docs.oggm.org/en/latest/) applied to Greenland peripheral calving glaciers and produce the results of the paper submitted to Journal of Glaciology. *Recinos, B. et al (in review)*

This repository uses OGGMv1.2.0 pinned to the following [commit](https://github.com/OGGM/oggm/commit/d13b4438c6f0be2266cafb1ba21aa526eef93c14).

The contents of the repository are the following:

- `calibration_scripts`: Python scripts to find k values for different model configurations. Each configuration is constructed by finding the intercepts between model (Frontal ablation or surface velocity) estimates and velocity observations and RACMO Frontal ablation fluxes, including the intercepts to the lower and upper error. 
- `cluster_scripts`: OGGM runs to produce the data for the calibration scripts and final results. (To be run in a cluster environment).
- `k_tools`: Python modules to re-project velocity observations and RACMO data into the OGGM glacier grid. Also contain a variety of functions to assist with the calibration process.
- `config.ini`: Global paths to data input and output. 

> The documentation and calibration method is still under constant development. 

[![DOI](https://zenodo.org/badge/249556625.svg)](https://zenodo.org/badge/latestdoi/249556625)



