#!/usr/bin/env python3

"""
Run the main loop of the simulation
"""

import sys
import numpy as np

from boids import *
from statistics import StatisticsHelper

# Set these parameters -- TODO: argparse!
SEED = 2

DT = 0.2

RUN_TIME = np.inf # in units of time
DUMP_STATS_INTERVAL = 10 # in units of *iterations* (one iteration = 1 DT time)

WORLD_RADIUS = 500
NUM_PREY = 50
NUM_PREDATORS = 2


PREY_RADIUS = 2.0
PREY_PERCEPTION_LENGTH = 150.0
PREY_PERCEPTION_ANGLE = np.pi*3.0/4.0 # Can see 135° in both directions.
PREY_TOO_CLOSE_RADIUS = 5.0
PREY_MAX_STEERING_ANGLE = np.pi/4.0 # Can turn maximum 45° per time step.
PREY_MAX_VELOCITY = 5.0
PREY_MIN_VELOCITY = 0.5
PREY_MAX_FORCE = 7.0
PREY_WEIGHT = 1.0
PREY_LIFESPAN = 8000
PREY_COLLISION_RECOVERY_RATE = 0.05

PREDATOR_RADIUS = 2.0
PREDATOR_PERCEPTION_LENGTH = 150.0
PREDATOR_PERCEPTION_ANGLE = np.pi # Can see 180° in both directions, i.e. no limitation.
PREDATOR_TOO_CLOSE_RADIUS = 5.0
PREDATOR_MAX_STEERING_ANGLE = np.pi # Can turn 180° in both direction, i.e. no limitation.
PREDATOR_MAX_VELOCITY = 8.0
PREDATOR_MIN_VELOCITY = 0.5
PREDATOR_MAX_FORCE = 10.0
PREDATOR_WEIGHT = 1.0
PREDATOR_LIFESPAN = 2000
PREDATOR_COLLISION_RECOVERY_RATE = 0.01

WEIGHTS_DISTRIBUTION_STD = 0.2

FEEDING_AREA_RADIUS = 5.0
FEEDING_AREA_POSITION = (50, 50)

'''
Network weights. Predator:
1. Target prey position
2. Target prey velocity
3. Target pursuit 
4. Fellow predator position
5. Fellow predator velocity
6. Fellow predator too close
7. Perimeter sensor

Prey:
1. Fellow prey position
2. Fellow prey velocity
3. Fellow prey "too close"
4. Predator
5. Feeding area sensor
6. Perimeter sensor
'''

PREDATOR_NETWORK_WEIGHTS = 0.5*np.array([np.random.random(), np.random.random(),
                                     0.0, 0.0,
                                     -np.random.random()])
PREY_NETWORK_WEIGHTS = 0.5*np.array([np.random.random(), np.random.random(),
                                   -np.random.random(), -np.random.random(),
                                    0])


def export_stats(prey_statistics, predator_statistics, ecosystem):
    prey_statistics.update_data(ecosystem.prey, ecosystem.prey_tree)
    prey_statistics.export()
    predator_statistics.update_data(ecosystem.predators,
                                    ecosystem.predator_tree)
    predator_statistics.export()


def main():    
    np.random.seed(SEED)

    ecosystem = Ecosystem(WORLD_RADIUS, NUM_PREY, NUM_PREDATORS,
                          PREY_RADIUS, PREDATOR_RADIUS,
                          PREY_MAX_VELOCITY, PREDATOR_MAX_VELOCITY,
                          PREY_MIN_VELOCITY, PREDATOR_MIN_VELOCITY,
                          PREY_MAX_STEERING_ANGLE, PREDATOR_MAX_STEERING_ANGLE,
                          PREY_MAX_FORCE, PREDATOR_MAX_FORCE,
                          PREY_PERCEPTION_LENGTH, PREDATOR_PERCEPTION_LENGTH,
                          PREY_PERCEPTION_ANGLE, PREDATOR_PERCEPTION_ANGLE,
                          PREY_TOO_CLOSE_RADIUS, PREDATOR_TOO_CLOSE_RADIUS,
                          PREY_NETWORK_WEIGHTS, PREDATOR_NETWORK_WEIGHTS,
                          PREY_LIFESPAN, PREDATOR_LIFESPAN,
                          PREY_WEIGHT, PREDATOR_WEIGHT,
                          PREY_COLLISION_RECOVERY_RATE, PREDATOR_COLLISION_RECOVERY_RATE,
                          FEEDING_AREA_RADIUS, FEEDING_AREA_POSITION, DT,
                          WEIGHTS_DISTRIBUTION_STD)

    prey_statistics = StatisticsHelper('prey_', False,
                                       True, True, True, True)
    predator_statistics = StatisticsHelper('predator_', False,
                                           True, True, True, True)

    t = 0
    iteration = 0
    while t < RUN_TIME:
        sys.stdout.write("\rt = %.1f" % t)
        sys.stdout.flush()

        # Collision check
        #  -- predators collided with prey are marked as feeding
        #  -- prey collided with feeding area marked as feeding
        #  -- collisions induce velocity change

        # Feeding
        #  -- those boids marked as feeding receive a linear(?) increase in stamina
        ecosystem.update_stamina()

        # Kill and spawn
        #  -- check for boids marked as dead, and respawn them
        #  -- update lifespan values for living boids
        ecosystem.kill_prey()
        ecosystem.kill_predators()
        ecosystem.update_age()

        # Compute statistics and dump data to disk
        if not iteration % DUMP_STATS_INTERVAL:
            export_stats(prey_statistics, predator_statistics, ecosystem)

        # Escape criterion: if avg life span not increasing for large no. of runs

        # Update time
        t += DT
        iteration += 1

        # Update positions (and corresponding regions) and stamina
        ecosystem.update_boid_velocities()
        ecosystem.update_velocity_data()
        ecosystem.update_boid_positions()
        ecosystem.update_position_data()

        # ...

    sys.stdout.write("\n")

if __name__ == '__main__':
    main()    
