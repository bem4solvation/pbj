"""Poisson Boltzmann & Jupyter: Bempp based biomolecular electrostatics solver."""

# Add imports here
from .electrostatics import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
