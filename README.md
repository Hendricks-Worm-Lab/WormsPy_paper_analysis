README: Data from: An epifluorescence microscope design for naturalistic behavior and cellular activity in freely moving Caenorhabditis elegans
[INSERT DRYAD DOI HERE] (bioRxiv pre-print: https://doi.org/10.1101/2025.03.21.644605)

# Wittekindt et al. Analysis Code
Analysis code published in Wittekindt et al., 2025 (https://www.biorxiv.org/content/10.1101/2025.03.21.644605v1)

## Description of the data and file structure

This data repository contains scripts and analysis pipelines for the paper "An epifluorescence microscope design for naturalistic behavior and cellular activity in freely moving Caenorhabditis elegans". The repository is split into four primary directories based on the tissue/neuron analyzed:

- **ASH/**: Contains scripts for analyzing ASH neuronal calcium imaging data (dual-channel GCaMP/RFP). Includes processing steps like segmentation and Two-channel Motion Artifact Correction (TMAC).
- **AWC/**: Contains scripts for analyzing single-channel AWC neuronal calcium imaging data and integration with behavioral annotation using DeepLabCut.
- **Muscle/**: Contains scripts for analyzing muscle calcium activity along the body wall of *C. elegans* using midline-based segmentation and orthogonal intensity profiling.
- **RIA/**: Contains scripts for analyzing RIA interneuron calcium activity using SAM2-based segmentation and polar coordinate transformation for behavioral correlation.

Each directory contains its own `README.md` file with specific instructions on the required packages, processing pipeline order, and data structures. Data and scripts are primarily written in Python.

Sharing/Access information

This dataset, as well as other information necessary to reproduce the results in the associated publication, are available on GitHub. We recommend that the dataset is accessed through the following link:
https://github.com/Hendricks-Worm-Lab/WormsPy_paper_analysis

 Code/Software

https://github.com/Hendricks-Worm-Lab/WormsPy_paper_analysis
