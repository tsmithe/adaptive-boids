# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

@author: rikardvinge
"""

import numpy as np

SEED = 0
np.random.seed(SEED)

class Boid:
    """
    Variables
    - Position, 1x2 array
    - Velocity, 1x2 array
    - Network weights, 1xn array
    - Stamina, double
    - Eating, boolean
    - Region/tree?
    - Age, integer
    - Maximum speed, double
    
    Class functions
    - Update position
    - Update velocity
    - Mutate network (for offspring)
    - Visible neighbours
    
    Property functions
    - sensors, nx2 array
    - acceleration, 1x2 array
    - killed (Check if living), boolean
    - 
    
    """
    def __init__(self, world_size):
        self.position = np.random.random(2)*world_size
        self.velocity = np.random.random(2) # TODO: decide range
        self.stamina = 1.0 # in range [0, 1] ?
        self.eating = False
        self.age = 0
        self.creep_range = 0.1 # how large?





    @property
    def sensors(self):
        raise NotImplementedError("Should be specialised by the subclass")

    @property
    def acceleration(self):
        """
        Call neural network with self.sensors as input to compute acceleration

        Doesn't need to be specialised by the subclass, since the computation
          is effectively the same
        """
        
        return self.sensors[0,:] # use neural work instead!

    def update_velocity(self, dt):
        """
        Update velocity by calling acceleration property.
        If speed is higher than maximum_speed, then decrease speed to 
        maximum_speed but keep direction.
        """
        self.velocity += self.acceleration * dt
        current_speed = np.sqrt(np.dot(self.velocity,self.velocity))
        if (current_speed > self.maximum_speed):
            self.velocity *= self.maximum_speed/current_speed

    def update_position(self, dt):
        self.update_velocity(dt)
        self.position += self.velocity * dt

    def mutate(self):
        """
        Mutate neural network weights.
        Implemented as a linearly distributed creep mutation.
        Returns a network with weights mutated with linear creep within
        [-creep_range,creep_range] from the original weights.
        No upper or lower bounds.
        """
        network_size = np.size(self.weights)
        mutated_weights = (self.weights.copy() - 
            2*self.creep_range*(np.random.random(network_size)-0.5))
        return mutated_weights
        
    def find_neighbours(self, tree, radius):
        """
        Takes cKDTree as input and finds all neighbours within radius.
        Input the cKDTree containing position of boids, obstacles or regions
        and the radius of the lookup.
        Returns the indices, in the cKDTree, of the objects with centers that
        are within radius from the center, including itself.
        
        Input radius can be perception_limit, too-close-limit or other.
        Use to calculate sensor values and check for collisions.
        """
        neighbour_indices = tree.query_ball_point(self.position, radius)
        return neighbour_indices
        
    def find_visible_neighbours(self, tree):
        """
        Takes a cKDTree as input and finds all neighbours within a circle 
        of radius perception_length.
        Checks if each neighbour is within the perception_angle.
        Returns indices of the visible objects.
        
        Calls self.find_neighbours(tree, perception_length) to find all
        neighbour indices. Removes any object located at the exact same
        position as the search center, e.g. a prey won't find itself.
        """
        neighbour_index = self.find_neighbours(tree, self.perception_length)
        visible_neighbours_index = []
        for i in neighbour_index:
            relative_position = tree.data[i,:] - self.position
            angle = np.arccos(np.dot(relative_position,self.velocity)/(
                np.sqrt(np.dot(relative_position,relative_position))*
                np.sqrt(np.dot(self.velocity,self.velocity))))
            if (~np.isnan(angle) & (np.abs(angle) < self.perception_angle)):
                visible_neighbours_index.append(i)
        return visible_neighbours_index
        
    @property
    def killed(self):
        return False

class Prey(Boid):
    def __init__(self, world_size):
        Boid.__init__(self, world_size) # call the Boid constructor, too

        self.number_of_weights = 5
        self.weights = np.random.random(self.number_of_weights) # neural net weights
        self.maximum_speed = 1 # How large?
        self.boid_radius = 1 # How large?
        self.perception_length = 2 # How large?
        self.perception_angle = np.pi/2 # how large? Should it differ between prey/predators.
        self.prey_tree = [] # Should this be initialized as an empty cKDTree?
        self.prey_flock_velocities = []

    @property
    def sensors(self):
#    def sensors(self, prey_tree, predator_tree):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
        
        This function currently assumes prey_tree and predator_tree are of
        the class cKDTree, NOT KDTree!

        Do something different from Prey.sensors!
        """
        # Find visible prey
        visible_prey_index = self.find_visible_neighbours(self.prey_tree)
        number_of_visible_prey = np.size(visible_prey_index)
        
        # Calculate prey-prey position sensor value.
        visible_prey_positions = self.prey_tree.data[visible_prey_index,:]
        prey_position_sensor = (
            np.sum(visible_prey_positions, axis=0)/number_of_visible_prey)
            
        # Calculate prey-prey velocity sensor value.
        visible_prey_velocities = (
            self.prey_flock_velocities[visible_prey_index,:])
        prey_velocity_sensor = (
            np.sum(visible_prey_velocities, axis=0)/number_of_visible_prey)            
        
        return np.random.random(
            2*self.number_of_weights).reshape(self.number_of_weights,2)

    @property
    def killed(self):
        """
        Return a boolean value describing whether boid is dead
        """
        death_probability = 1 - self.stamina
        if np.random.random() < death_probability:
            return False
        else:
            return True

class Predator(Boid):
    def __init__(self, world_size):
        Boid.__init__(self, world_size) # call the Boid constructor, too

        self.number_of_weights = 6
        self.weights = np.random.random(self.number_of_weights) # neural net weights
        self.maximum_speed = 1 # how large?
        self.boid_radius = 2 # How large?
        self.perception_angle = np.pi/2 # how large? Should it differ between prey/predators.

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2

        Do something different from Prey.sensors!
        """
        return np.random.random(
            2*self.number_of_weights).reshape(self.number_of_weights,2)
