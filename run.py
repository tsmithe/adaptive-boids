"""
Run the main loop of the simulation
"""

import sys
import numpy as np
from boids import *

SEED = 0
np.random.seed(SEED)

while True:
    # Escape criterion: if avg life span not increasing for large no. of runs

    # Update time
    
    # Collision check
    #  -- predators collided with prey are marked as feeding
    #  -- prey collided with feeding area marked as feeding
    #  -- collisions induce velocity change

    # Feeding
    #  -- those boids marked as feeding receive a linear(?) increase in stamina

    # Kill and spawn
    #  -- check for boids marked as dead, and respawn them
    #  -- update lifespan values for living boids
    
    # Compute statistics and dump data to disk

    # Update positions (and corresponding regions) and stamina

    # ...
    pass

