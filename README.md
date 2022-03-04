Poisson Boltzmann & Jupyter: Python based biomolecular electrostatics solver.
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/pbj/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/pbj/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PBJ/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PBJ/branch/master)


Welcome! PBJ may not also be the acronym of your favorite sandwich, but also your favorite Poisson-Boltzmann solver. Poisson-Boltzmann and Jupyter (PBJ) is a boundary element solver based on the [Bempp-cl library](https://bempp.com/). It provides an easy API for electrostatics calculations from a Jupyter notebook, with full access to all the nice (and fast!) features in Bempp-cl. It is built to be easily extensible, like a playground for new models!

Installation
================
Download the repository and create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) using the `environment.yml` file as

`conda env create -f environment.yml`

and you're good to go!

Documentation
===============
This is work-in-progress, but you'll be able to find examples under `notebooks/tutorials/`.

The team
==============
This code is developed by a [research group](https://bem4solvation.github.io/) based in the Mechanical Engineering Department of [Universidad Técnica Federico Santa María](https://usm.cl/). The main contributors are

* Stefan Search (@sdsearch): initial design and main functions
* Kenneth Styles (@kstylesc): implementation of first version and the SLIC model
* Miguel Godoy (@mgodoydiaz): implementation of firt version
* Sergio Urzúa (@urzuasergio): protein surface interacion features (under development)
* Ian Addison-Smith (@iaddison-smith): force calculation (under development)
* Christopher Cooper (@cdcooper84): PI

How to cite
==============
Search, S. D., Cooper, C. D., van't Wout, E., "Towards optimal boundary integral formulations of the Poisson–Boltzmann equation for molecular electrostatics" J. Comput. Chem. 2022, 1. https://doi.org/10.1002/jcc.26825. [Preprint on arXiv](https://arxiv.org/abs/2108.10481) 


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
