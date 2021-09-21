"""Poisson Boltzmann & Jupyter: Bempp based biomolecular electrostatics solver."""

# Add imports here
import bempp.api
import os
from pbj.electrostatics.solute import *
from pbj.nonpolar import * 
from pbj.mesh import *
# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
