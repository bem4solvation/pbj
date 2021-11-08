import bempp.api
import os 
from pbj.electrostatics.solute import *
from .solute import *
from .pb_formulation import * ######

PBJ_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))