"""Poisson Boltzmann & Jupyter: Bempp based biomolecular electrostatics solver."""
# Add imports here
import os
from pbj.electrostatics.solute import Solute

PBJ_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
