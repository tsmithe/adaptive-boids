#!/usr/bin/env python3

"""
Run the main loop of the simulation
"""

import sys
import numpy as np

from boids import *
from statistics import StatisticsHelper

# Set these parameters -- TODO: argparse!
SEED = 0
DT = 1
RUN_TIME = 1000
DUMP_STATS_INTERVAL = 10
WORLD_SIZE = 10
NUM_PREY = 200
NUM_PREDATORS = 20
#TODO: feeding area params?
FEEDING_AREA_POSITION = (2, 2)

np.random.seed(SEED)

ecosystem = Ecosystem(WORLD_SIZE, NUM_PREY, NUM_PREDATORS,
                      FEEDING_AREA_POSITION, DT)

def export_stats(ecosystem):
    prey_statistics = StatisticsHelper(ecosystem.prey, ecosystem.prey_tree,
                                       'prey_',
                                       True, True, True)
    predator_statistics = StatisticsHelper(ecosystem.predators,
                                           ecosystem.predator_tree,
                                           'predator_',
                                           True, True, True)
    prey_statistics.export()
    predator_statistics.export()
    

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
    if not t % DUMP_STATS_INTERVAL:
        export_stats(ecosystem)

    # Escape criterion: if avg life span not increasing for large no. of runs

    # Update time
    t += DT

    # Update positions (and corresponding regions) and stamina
    ecosystem.update_velocities()
    ecosystem.update_positions()

    # ...

sys.stdout.write("\n")
