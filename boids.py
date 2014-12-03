# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

@author: rikardvinge
"""

import numpy as np
from scipy.spatial import cKDTree
import random

class Ecosystem:
    def __init__(self, world_size, num_prey, num_predators,
                 prey_radius, predator_radius,
                 feeding_area_radius, feeding_area_position, dt):
        self.dt = dt
        self.world_size = world_size
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.feeding_area_position = np.asarray(feeding_area_position)
        self.feeding_area_radius = feeding_area_radius

        self.prey_radius = prey_radius
        self.predator_radius = predator_radius
        
        self.prey = []
        self.predators = []

        for i in range(self.num_prey):
            self.prey.append(Prey(self))

        for i in range(self.num_predators):
            self.predators.append(Predator(self))

        self.update_position_data()
        self.update_velocity_data()

    def update_position_data(self):
        self.prey_positions = np.array([p.position for p in self.prey])
        self.predator_positions = np.array([p.position for p in self.predators])
        self.prey_tree = cKDTree(self.prey_positions)
        self.predator_tree = cKDTree(self.predator_positions)

    def update_boid_positions(self):
        for b in self.predators + self.prey:
            b.update_position(self.dt)
        #self.all_positions = np.concatenate(prey_positions, predator_positions)

    def update_velocity_data(self):
        self.prey_velocities = np.array([p.velocity for p in self.prey])
        self.predator_velocities = np.array([p.velocity for p in self.predators])

    def update_boid_velocities(self):
        for b in self.predators + self.prey:
            b.update_velocity(self.dt)
        
    def update_stamina(self):
        for b in self.prey:
            if (np.linalg.norm(b.position-self.feeding_area_position)<
                self.feeding_area_radius):
                b.eating = True
                if b.stamina < 1:
                    b.stamina += 0.01*self.dt
            else:
                b.eating = False
                if b.stamina > 0:
                    b.stamina -=0.01*self.dt

    def update_age(self):
        for b in self.predators + self.prey:
            b.age += self.dt

    def roulette_selection(self,weights):
        total = 0
        winner = 0
        for i, w in enumerate(weights):
            total += w
            if random.random() * total < w:
                winner = i
        return winner

    def kill_prey(self):
        # Kill prey
        for b in self.prey:
            if b.killed == True:
                self.prey.remove(b)
                
        # Replace dead prey
        dead_prey = self.num_prey-len(self.prey)
        if dead_prey > 0:
            fitness_values = [1/(b.age+0.01) for b in self.prey]
            parent = self.roulette_selection(fitness_values)
            child = Prey(self)
            child.weights = self.prey[parent].weights
            child.weights = child.mutate()
            
            for i in range(dead_prey):
                self.prey.append(child)

class Boid:

    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.stamina = 1.0 # in range [0, 1] ?
        self.eating = False
        self.age = 0
        self.creep_range = 1
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
        if np.linalg.norm(self.position) > self.ecosystem.world_size:
            return self.sensors/self.boid_weight - self.position *np.exp( # A suitable scaling parameter is needed
                (np.linalg.norm(self.position)-self.ecosystem.world_size))/np.linalg.norm(self.position)
        else:
            return self.sensors/self.boid_weight # use neural work instead!

    def update_velocity(self, dt):
        self.velocity += self.acceleration * dt

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
            angle = np.abs(np.arccos(np.dot(relative_position,self.velocity)/(
                np.sqrt(np.dot(relative_position,relative_position))*
                np.sqrt(np.dot(self.velocity,self.velocity)))))
            if (not np.isnan(angle) and (np.abs(angle) < self.perception_angle)):
                visible_neighbours_index.append(i)
        return visible_neighbours_index
        
    @property
    def killed(self):
        return False

class Prey(Boid):
    def __init__(self, ecosystem):
        Boid.__init__(self, ecosystem) # call the Boid constructor, too
        
        self.number_of_weights = 5
        self.maximum_speed = 1 # How large?
        self.minimum_speed = 0.1 # How small?
        self.perception_length = 2 # How large?
        self.perception_angle = np.pi/2 # How large? Should it differ between prey/predators.
        self.too_close_radius = 1 # How large?
        self.boid_weight = 1 # How large?
        self.life_span = 200 # how large?

        self.weights = 2*np.random.random(self.number_of_weights)-1 # neural net weights        
        self.position = ecosystem.world_size*(2*np.random.random(2)-1)
        self.velocity = (self.maximum_speed/np.sqrt(2))*(2*np.random.random(2)-1) # TODO: decide range

    def update_velocity(self, dt):
        """
        Checks if prey has collided with other prey. We don't have to check if
        has collided with predator since this is done in property killed.
        If colided: keep current flight direction and set speed to self.minimum_speed.
        Else: calculate new acceleration and new velocity.
        If new velocity > self.maximum_speed: Reduce speed to self.maximum_speed
        """
        collided_with = self.ecosystem.prey_tree.query_ball_point(self.position,
            2*self.ecosystem.prey_radius)
        
        if len(collided_with) > 0:
            self.velocity = self.velocity/np.linalg.norm(self.velocity)*self.minimum_speed
        else:
            self.velocity += self.acceleration * dt
            
        if np.linalg.norm(self.velocity) > self.maximum_speed:
            self.velocity = self.velocity*(self.maximum_speed/np.linalg.norm(self.velocity))

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
            self.ecosystem.prey_tree, self.perception_length)
        number_of_visible_prey = np.size(visible_prey_index)

        if (number_of_visible_prey > 0):        
            # Calculate fellow prey position sensor value.
            relative_prey_positions = (self.ecosystem.prey_tree.data[visible_prey_index,:] - 
                self.position)
            # Calculate fellow prey velocity sensor value.
            relative_prey_velocities = (
                self.prey_velocities[visible_prey_index,:]-self.velocity)
            if (number_of_visible_prey == 1):
                sensors[0,:] = relative_prey_positions
                sensors[1,:] = relative_prey_velocities
            else:
                sensors[0,:] = (np.sum(relative_prey_positions, axis=0)/
                    number_of_visible_prey)
                sensors[1,:] = (np.sum(relative_prey_velocities, axis=0)/
                    number_of_visible_prey)

            
        # Find "too close" prey.
        too_close_index = self.find_visible_neighbours(
            self.ecosystem.prey_tree, self.too_close_radius)
        number_of_too_close = np.size(too_close_index)
            
        if (number_of_too_close > 0):
            # Calculate "too close" sensor value.
            relative_too_close_positions = np.array(
                self.ecosystem.prey_tree.data[too_close_index,:] - self.position)
            if (number_of_too_close == 1):
                too_close_distance = np.linalg.norm(relative_too_close_positions)
                too_close_direction = relative_too_close_positions/too_close_distance
            else:
                too_close_distance = np.linalg.norm(relative_too_close_positions, axis=1)
                too_close_direction = (
                    relative_too_close_positions/too_close_distance[:,np.newaxis])
            sensors[2,:] = (np.dot(((self.too_close_radius/too_close_distance) - 1),
                too_close_direction)/number_of_too_close)

        # Find visible predators.
        visible_predator_index = self.find_visible_neighbours(
            self.ecosystem.predator_tree, self.perception_length)
        number_of_visible_predators = np.size(visible_predator_index)

        if (number_of_visible_predators > 0):
            # Calculate predator sensor.
            relative_predator_positions = np.array(
                self.ecosystem.predator_tree.data[visible_predator_index,:] - self.position)
            if (number_of_visible_predators == 1):
                distance_to_predator = np.linalg.norm(relative_predator_positions)
                direction_to_predator = (relative_predator_positions/distance_to_predator)
            else:
                distance_to_predator = np.linalg.norm(relative_predator_positions,axis=1)
                direction_to_predator = (
                    relative_predator_positions/distance_to_predator[:,np.newaxis])
            sensors[3,:] = (np.dot(((self.perception_length/distance_to_predator) - 1),
                direction_to_predator)/number_of_visible_predators)
            
        # Feeding area sensor, assuming only one area and perfect vision.
        relative_feeding_position = (self.ecosystem.feeding_area_position-self.position)
        sensors[4,:] = np.zeros(2)#relative_feeding_position
         
        force = np.dot(self.weights,sensors)
        
        return force

    @property
    def killed(self):
        """
        Return a boolean value describing whether boid is dead.
        Prey can die of collision with predator and old age (for now).
        """
        collided_with = self.ecosystem.predator_tree.query_ball_point(self.position,
            self.ecosystem.prey_radius+self.ecosystem.predator_radius)
        
        if len(collided_with) > 0:
            return True
        elif (self.age > self.life_span):
            return True
        else:
            return False
        """
        death_probability = (1 - self.stamina)
        if np.random.random() < death_probability:
            return False
        else:
            return True
        """

class Predator(Boid):
    def __init__(self, ecosystem):
        Boid.__init__(self, ecosystem) # call the Boid constructor, too

        self.number_of_weights = 5
        self.maximum_speed = 1 # how large?
        self.minimum_speed = 0.1 # How small?
        self.perception_length = 5 # How large?
        self.perception_angle = np.pi*3/2 # how large? Should it differ between prey/predators.
        self.too_close_radius = 3 # How large?
        self.predator_velocities = []
        self.boid_weight = 1 # How large?
        self.life_span = 200 # How large?
        
        self.weights = 2*np.random.random(self.number_of_weights)-1 # neural net weights
        self.position = ecosystem.world_size*(2*np.random.random(2)-1)
        self.velocity = (self.maximum_speed/np.sqrt(2))*(2*np.random.random(2)-1)

    def update_velocity(self, dt):
        """
        Checks if prey has collided with other prey. We don't have to check if
        has collided with predator since this is done in property killed.
        If colided: keep current flight direction and set speed to self.minimum_speed.
        Else: calculate new acceleration and new velocity.
        If new velocity > self.maximum_speed: Reduce speed to self.maximum_speed
        """
        collided_with = self.ecosystem.predator_tree.query_ball_point(self.position,
            2*self.ecosystem.predator_radius)
        
        if len(collided_with) > 0:
            self.velocity = self.velocity/np.linalg.norm(self.velocity)*self.minimum_speed
        else:
            self.velocity += self.acceleration * dt
            
        if np.linalg.norm(self.velocity) > self.maximum_speed:
            self.velocity = self.velocity*(self.maximum_speed/np.linalg.norm(self.velocity))


    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
        """
        sensors = np.zeros([self.number_of_weights,2])
        
        # Find visible prey.
        visible_prey_index = self.find_visible_neighbours(
            self.ecosystem.prey_tree, self.perception_length)
        number_of_visible_prey = np.size(visible_prey_index)
    
        # Target prey position sensor and target velocity. Chooses the prey 
        # that is closest to the predator.        
        if (number_of_visible_prey > 0):
            relative_prey_positions = np.array(
                self.ecosystem.prey_tree.data[visible_prey_index,:] - self.position)
            prey_distance = np.linalg.norm(relative_prey_positions, axis=1)
            target_prey_index = visible_prey_index[np.argmin(prey_distance)]
            # !!! TODO
            sensors[0,:] = self.ecosystem.prey_tree.data[target_prey_index,:]
            sensors[1,:] = self.ecosystem.prey_velocities[target_prey_index,:]

        
        # Find visible predators.
        visible_predator_index = self.find_visible_neighbours(
            self.ecosystem.predator_tree, self.perception_length)
        number_of_visible_predators = np.size(visible_predator_index)
        
        # Fellow predator position and velocity sensor values.
        if (number_of_visible_predators > 0):
            relative_predator_positions = np.array(
                self.ecosystem.predator_tree.data[visible_predator_index,:]-self.position)
            relative_predator_velocities = np.array(
                self.ecosystem.predator_velocities[visible_predator_index]-self.velocity)
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
            self.ecosystem.predator_tree, self.too_close_radius)
        number_of_too_close = np.size(too_close_index)
        
        # Calculate "too close" sensor value.
        if (number_of_too_close > 0):
            relative_too_close_positions = np.array(
                self.ecosystem.predator_tree.data[too_close_index,:] - self.position)
            if (number_of_too_close == 1):
                too_close_dist = np.linalg.norm(relative_too_close_positions)
                too_close_direction = (relative_too_close_positions/too_close_dist)
            else:
                too_close_dist = np.linalg.norm(relative_too_close_positions, axis=1)
                too_close_direction = (
                    relative_too_close_positions/too_close_dist[:,np.newaxis])
            sensors[4,:] = (np.dot(((self.too_close_radius/too_close_dist)-1),
                too_close_direction)/number_of_visible_predators)
    
        # Total force.           
        force = np.dot(self.weights,sensors)
        
        return force
        
    @property
    def killed(self):
        """
        Return a boolean value describing whether boid is dead.
        Predators can only dies of old age (for now).
        """

        if (self.age > self.life_span):
            return True
        else:
            return False
