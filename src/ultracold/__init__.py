"""
Ultracold atoms experiment analysis and simulation package.

Modules
-------
simulation
    Cloud dynamics and free fall simulation
analysis  
    Image analysis, profile extraction, and fitting
utils
    Physical constants and experimental parameters
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Convenient imports
from .utils.constants import (
    M_RB87, KB, HBAR, G,
    STATE_F2_MF2, STATE_F2_MF1,
    ExperimentParams, DEFAULT_PARAMS
)

from .simulation.cloud import ThermalCloud, MixedStateCloud
from .simulation.free_fall import simulate_free_fall, create_experiment_cloud
