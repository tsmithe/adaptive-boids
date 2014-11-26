"""
Run the main loop of the simulation
"""

import sys
import numpy as np
from boids import *

SEED = 0
np.random.seed(SEED)

# Lists to hold prey and predators
prey = []
predators = []

while True:
    # Collision check
    #  -- predators collided with prey are marked as feeding
    #  -- prey collided with feeding area marked as feeding
    #  -- collisions induce velocity change
    prey_positions = np.array([p.position for p in prey])
    predator_positions = np.array([p.position for p in prey])
    all_positions = np.concatenate(prey_positions, predator_positions)

    # Feeding
    #  -- those boids marked as feeding receive a linear(?) increase in stamina

    # Kill and spawn
    #  -- check for boids marked as dead, and respawn them
    #  -- update lifespan values for living boids
    
    # Compute statistics and dump data to disk

    # Escape criterion: if avg life span not increasing for large no. of runs

    # Update time

    # Update positions (and corresponding regions) and stamina

    # ...
    pass

