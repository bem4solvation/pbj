import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
import pbj
import pbj.implicit_solvent.pb_formulation.formulations as pb_formulations
from pbj.implicit_solvent.utils.analytical import an_P
from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -qq test_verification.py -s

# dos esferas, multiple superficies + fuerzas + energia  

# potencial con 1 esfera + fuerza (validar la herramienta de visualizacion de potencial)
# potencial en distintos puntos ver paper martin figura 6a), tomar 6 puntos algunos adentro y otras afuera


# solucion analitica, energia,m repo  