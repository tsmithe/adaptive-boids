# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

@author: rikardvinge
"""

import numpy as np
from scipy import spatial

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
        self.mutation_probability = 0.5





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
        
        return self.sensors/self.boid_weight # use neural work instead!

    def update_velocity(self, dt):
        """
        Update velocity by calling acceleration property.
        If speed is higher than maximum_speed, decrease speed to 
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
        Each weight is mutated with probability mutation_probability.
        Implemented as a linearly distributed creep mutation.
        Returns a network with weights mutated with linear creep within
        [-creep_range,creep_range] from the original weights.
        No upper or lower bounds.
        """
        network_size = np.size(self.weights)
        mutated_weights = self.weights.copy()
        for i in np.arange(network_size):
            if (np.random.random() < self.mutation_probability):
                mutated_weights[i] = (mutated_weights[i] - 
                    2*self.creep_range*(np.random.random()-0.5))            
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
        
    def find_visible_neighbours(self, tree, radius):
        """
        Takes a cKDTree as input and finds all neighbours within a circle 
        of radius "radius".
        Checks if each neighbour is within the perception_angle.
        Returns indices of the visible objects.
        
        Calls self.find_neighbours(tree, perception_length) to find all
        neighbour indices. Removes any object located at the exact same
        position as the search center, e.g. a prey won't find itself.
        """
        neighbour_index = self.find_neighbours(tree, radius)
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
        self.perception_angle = np.pi/2 # How large? Should it differ between prey/predators.
        self.too_close_radius = 1 # How large?
        self.prey_tree = [] # Should this be initialized as an empty cKDTree?
        self.prey_flock_velocities = np.array([])
        self.predator_tree = []
        self.feeding_area_position = np.array([])
        self.boid_weight = 1 # How large?

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
        
        This function currently assumes prey_tree and predator_tree are of
        the class cKDTree, NOT KDTree!

        Do something different from Prey.sensors!
        """
        sensors = np.zeros([self.number_of_weights,2])
        
        # Find visible prey.
        visible_prey_index = self.find_visible_neighbours(
            self.prey_tree, self.perception_length)
        number_of_visible_prey = np.size(visible_prey_index)

        # Calculate fellow prey position and velocity sensor values.
        if (number_of_visible_prey > 0):        
            relative_prey_positions = (
                self.prey_tree.data[visible_prey_index,:] - self.position)
            relative_prey_velocities = (
                self.prey_flock_velocities[visible_prey_index,:]-self.velocity)
            if (number_of_visible_prey == 1):
                sensors[0,:] = relative_prey_positions
                sensors[1,:] = relative_prey_velocities
            else:
                sensors[0,:] = (np.sum(
                    relative_prey_positions, axis=0)/number_of_visible_prey)
                sensors[1,:] = (np.sum(
                    relative_prey_velocities,axis=0)/number_of_visible_prey)

        # Find "too close" prey.
        too_close_index = self.find_visible_neighbours(
            self.prey_tree, self.too_close_radius)
        number_of_too_close = np.size(too_close_index)
  
        # Calculate "too close" sensor value.
        if (number_of_too_close > 0):
            relative_too_close_positions = np.array(
                self.prey_tree.data[too_close_index,:] - self.position)
            too_close_dist = np.linalg.norm(
                relative_too_close_positions, axis=1)
            too_close_direction = (
                relative_too_close_positions/too_close_dist[:,np.newaxis])
            sensors[2,:] = (np.dot(((self.too_close_radius/too_close_dist)-1),
                too_close_direction)/number_of_too_close)

        # Find visible predators.
        visible_predator_index = self.find_visible_neighbours(
            self.predator_tree, self.perception_length)
        number_of_visible_predators = np.size(visible_predator_index)

        # Calculate predator sensor.
        if (number_of_visible_predators > 0):
            relative_predator_positions = np.array(
                self.predator_tree.data[visible_predator_index,:]-self.position)
            dist_to_predator = np.linalg.norm(
                relative_predator_positions, axis=1)
            direction_to_predator = np.array(
                relative_predator_positions/dist_to_predator[:,np.newaxis])
            sensors[3,:] = (np.dot(
                ((self.perception_length/dist_to_predator)-1),
                direction_to_predator)/number_of_visible_predators)

            
        # Feeding area sensor, assuming only one area and perfect prey vision.
        relative_feeding_position = np.array(
            self.feeding_area_position-self.position)
        sensors[4,:] = relative_feeding_position
        
        # Total force.
        force = np.dot(self.weights,sensors)
        
        return force

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

        self.number_of_weights = 5
        self.weights = np.random.random(self.number_of_weights) # neural net weights
        self.maximum_speed = 1 # how large?
        self.boid_radius = 2 # How large?
        self.perception_length = 1 # How large?
        self.perception_angle = np.pi/2 # how large? Should it differ between prey/predators.
        self.too_close_radius = 1 # How large?
        self.prey_tree = [] # Should this be initialized as an empty cKDTree?
        self.prey_flock_velocities = []
        self.predator_tree = []
        self.predator_velocities = []
        self.boid_weight = 1 # How large?

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2

        Do something different from Prey.sensors!
        """
        sensors = np.zeros([self.number_of_weights,2])
        
        # Find visible prey.
        visible_prey_index = self.find_visible_neighbours(
            self.prey_tree, self.perception_length)
        number_of_visible_prey = np.size(visible_prey_index)

        # Target prey position sensor and target velocity. Chooses the prey 
        # that is closest to the predator.        
        if (number_of_visible_prey > 0):
            relative_prey_positions = np.array(
                self.prey_tree.data[visible_prey_index,:] - self.position)
            prey_distance = np.linalg.norm(relative_prey_positions, axis=1)
            target_prey_index = visible_prey_index[np.argmin(prey_distance)]
            sensors[0,:] = self.prey_tree.data[target_prey_index,:]
            sensors[1,:] = self.prey_flock_velocities[target_prey_index,:]
        
        # Find visible predators.
        visible_predator_index = self.find_visible_neighbours(
            self.predator_tree, self.perception_length)
        number_of_visible_predators = np.size(visible_predator_index)
        
        # Fellow predator position and velocity sensor values.
        if (number_of_visible_predators > 0):
            relative_predator_positions = np.array(
                self.predator_tree.data[visible_predator_index,:]-self.position)
            relative_predator_velocities = np.array(
                self.predator_velocities[visible_predator_index,:]-self.velocity)
            if (number_of_visible_predators == 1):
                sensors[2,:] = relative_predator_positions
                sensors[3,:] = relative_predator_velocities
            else:
                sensors[2,:] = (np.sum(relative_predator_positions,axis=0)/
                    number_of_visible_predators)
                sensors[3,:] = (np.sum(relative_predator_velocities,axis=0)/
                    number_of_visible_predators)
                    
        # "too-close" predator sensor.
        too_close_index = self.find_visible_neighbours(
            self.predator_tree, self.too_close_radius)
        number_of_too_close = np.size(too_close_index)
        
        # Calculate "too close" sensor value.
        if (number_of_too_close > 0):
            relative_too_close_positions = np.array(
                self.predator_tree.data[too_close_index,:] - self.position)
            too_close_dist = np.linalg.norm(
                relative_too_close_positions, axis=1)
            too_close_direction = (
                relative_too_close_positions/too_close_dist[:,np.newaxis])
            sensors[4,:] = (np.dot(((self.too_close_radius/too_close_dist)-1),
                too_close_direction)/number_of_visible_predators)

        # Total force.           
        force = np.dot(self.weights,sensors)
        
        return force
