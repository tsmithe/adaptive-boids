#!/usr/bin/env python3

"""
Run the main loop of the simulation
"""

import sys
import numpy as np

from boids import *

# Set these parameters -- TODO: argparse!
SEED = 0
DT = 1
RUN_TIME = 1000
WORLD_SIZE = 10
NUM_PREY = 200
NUM_PREDATORS = 20
#TODO: feeding area params?
FEEDING_AREA_POSITION = (2, 2)

np.random.seed(SEED)

ecosystem = Ecosystem(WORLD_SIZE, NUM_PREY, NUM_PREDATORS,
                      FEEDING_AREA_POSITION, DT)
t = 0
while t < RUN_TIME:
    sys.stdout.write("\rt = %.1f" % t)
    sys.stdout.flush()
    
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

    # Escape criterion: if avg life span not increasing for large no. of runs

    # Update time
    t += DT

    # Update positions (and corresponding regions) and stamina
    ecosystem.update_velocities()
    ecosystem.update_positions()

    # ...

sys.stdout.write("\n")
