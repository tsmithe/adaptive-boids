# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 15:29:50 2014

@author: rikardvinge
"""

import numpy as np
from scipy import spatial
from boids import *

SEED = 0
np.random.seed(SEED)

# Lists to hold prey and predators
prey = []
predators = []

dt = 1
world_size = 10
number_of_prey = 200
number_of_predators = 20

# Create prey boids.
for i in np.arange(number_of_prey):
    prey.append(Prey(world_size))

for i in np.arange(number_of_predators):
    predators.append(Predator(world_size))
    
# Create array of prey and predator positions and velocities and create cKDTree.
prey_positions = np.array([p.position for p in prey])
prey_velocities = np.array([p.velocity for p in prey])
prey_tree = spatial.cKDTree(prey_positions)

predator_positions = np.array([p.position for p in predators])
predator_velocities = np.array([p.velocity for p in predators])
predator_tree = spatial.cKDTree(predator_positions)

# Feeding area
feeding_area_position = np.array([2,2])

# Update prey_tree variable in each prey (can it be done more efficiently?)
for p in prey:
    p.prey_tree = prey_tree
    p.prey_flock_velocities = prey_velocities
    p.predator_tree = predator_tree
    p.feeding_area_position = feeding_area_position
    
for p in predators:
    p.prey_tree = prey_tree
    p.prey_flock_velocities = prey_velocities
    p.predator_tree = predator_tree
    p.predator_velocities = predator_velocities
    
print(prey[1].position)    
prey[1].update_position(dt)
print(prey[1].position)    
print(prey[1].weights)
print(prey[1].mutate())

    
