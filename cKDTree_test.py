"""
Run the main loop of the simulation
"""

import sys
import numpy as np
import scipy as sci
from boids import *


SEED = 0
np.random.seed(SEED)

# Lists to hold prey and predators
prey = []
predators = []

dt = 1
world_size = 10
number_of_boids = 60

# Create prey boids.
for i in np.arange(number_of_boids):
    prey.append(Prey(world_size))
    
# Create array of prey positions and velocities and create cKDTree.
prey_positions = np.array([p.position for p in prey])
prey_velocities = np.array([p.velocity for p in prey])

prey_tree = sci.spatial.cKDTree(prey_positions)


# Update prey_tree variable in each prey (can it be done more efficiently?)
for p in prey:
    p.prey_tree = prey_tree
    p.prey_flock_velocities = prey_velocities
    
neighbours = prey[0].find_neighbours(prey[0].prey_tree, prey[0].perception_length)
visible_neighbours = prey[0].find_visible_neighbours(
    prey[0].prey_tree, prey[0].perception_length)
print('Indices of all neighbours within perception_length')
print(neighbours)
print('Indices of visible neighbours within boid perception_length')
print(visible_neighbours)
print('Positions of all visible neighbours')
print(prey[0].prey_tree.data[visible_neighbours,:])

# Update prey positions.
for p in prey:
    p.update_position(dt)

# Recreate the prey_position array and cKDTree.
prey_positions = np.array([p.position for p in prey])
prey_velocities = np.array([p.velocity for p in prey])

prey_tree = sci.spatial.cKDTree(prey_positions)

# Update prey_tree variable in each prey (can it be done more efficiently?)
for p in prey:
    p.prey_tree = prey_tree
    p.prey_flock_velocities = prey_velocities

neighbours = prey[0].find_neighbours(prey[0].prey_tree, prey[0].perception_length)
visible_neighbours = prey[0].find_visible_neighbours(
    prey[0].prey_tree, prey[0].perception_length)
neighbour_positions = prey[0].prey_tree.data[neighbours,:]
print('Indices of all neighbours after time step')
print(neighbours)
print('Indices of visible neighbours after time step')
print(visible_neighbours)
print('Positions of all visible neighbours after time step')
print(prey[0].prey_tree.data[visible_neighbours,:])

"""
Test whether indices in the prey_tree points to the same value as that index
prey_positions. Result: They do! Thus, we can keep an updated pointer to a 
prey_velocities array in the prey/predator classes and extract the velocities
of the visible prey using the indices found when searching for 
nearest-neighbours.
"""
print('Index test')
print(prey[0].prey_tree.data[visible_neighbours,:])
print(prey_positions[visible_neighbours,:])


