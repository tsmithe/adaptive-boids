# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

@author: rikardvinge
"""

import numpy as np
from scipy.spatial import cKDTree

import fast_boids
from fast_boids import quick_norm
"""
    ecosystem = Ecosystem(WORLD_RADIUS, NUM_PREY, NUM_PREDATORS,
                          PREY_RADIUS, PREDATOR_RADIUS,
                          PREY_MAX_VELOCITY, PREDATOR_MAX_VELOCITY,
                          PREY_MIN_VELOCITY, PREDATOR_MIN_VELOCITY,
                          PREY_PERCEPTION_LENGTH, PREDATOR_PERCEPTION_LENGTH,
                          PREY_PERCEPTION_ANGLE, PREDATOR_PERCEPTION_ANGLE,
                          PREY_TOO_CLOSE_RADIUS, PREDATOR_TOO_CLOSE_RADIUS,
                          PREY_NETWORK_WEIGHTS, PREDATOR_NETWORK_WEIGHTS,
                          PREY_LIFESPAN, PREDATOR_LIFESPAN,
                          PREY_WEIGHT, PREDATOR_WEIGHT,
                          FEEDING_AREA_RADIUS, FEEDING_AREA_POSITION, DT,
                          CREEP_RANGE, MUTATION_PROBABILITY,
                          NUMBER_OF_COLLISION_SPEED_INCREASE_STEPS,)
"""

class Ecosystem:
    def __init__(self, world_radius, num_prey, num_predators,
                 prey_radius, predator_radius,
                 prey_maximum_speed, predator_maxmimum_speed,
                 prey_minimum_speed, predator_minimum_speed,
                 prey_perception_length, predator_perception_length,
                 prey_perception_angle, predator_perception_angle,
                 prey_too_close_radius, predator_too_close_radius,
                 prey_network_weights, predator_network_weights,
                 prey_lifespan, predator_lifespan, 
                 prey_weight, predator_weight,
                 feeding_area_radius, feeding_area_position, dt,
                 creep_range, mutation_probability,
                 num_collision_speed_increase_steps):
        self.dt = dt
        self.world_radius = world_radius
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.feeding_area_position = np.asarray(feeding_area_position)
        self.feeding_area_radius = feeding_area_radius
        self.creep_range = creep_range
        self.mutation_probability = mutation_probability

        self.prey_radius = prey_radius
        self.prey_maximum_speed = prey_maximum_speed
        self.prey_minimum_speed = prey_minimum_speed
        self.prey_collision_speed_rebound = (self.prey_maximum_speed-self.prey_minimum_speed)/num_collision_speed_increase_steps
        self.prey_perception_length = prey_perception_length
        self.prey_perception_angle = prey_perception_angle
        self.prey_too_close_radius = prey_too_close_radius
        self.prey_weight = prey_weight
        self.prey_lifespan = prey_lifespan
        self.prey_network_weights = prey_network_weights

        self.predator_radius = predator_radius
        self.predator_maximum_speed = predator_maxmimum_speed
        self.predator_minimum_speed = predator_minimum_speed
        self.predator_collision_speed_rebound = (self.predator_maximum_speed-self.predator_minimum_speed)/num_collision_speed_increase_steps
        self.predator_perception_length = predator_perception_length
        self.predator_perception_angle = predator_perception_angle
        self.predator_too_close_radius = predator_too_close_radius
        self.predator_weight = predator_weight
        self.predator_lifespan = predator_lifespan
        self.predator_network_weights = predator_network_weights
        
        self.prey = []
        self.predators = []

        for i in range(self.num_prey):
            self.prey.append(Prey(self))

        for i in range(self.num_predators):
            self.predators.append(Predator(self))
            self.predators[i].life_span += i*self.dt

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
            if (quick_norm(b.position-self.feeding_area_position)<
                self.feeding_area_radius):
                b.eating = True
                if b.stamina < 1:
                    b.stamina += 0.01*self.dt
            else:
                b.eating = False
                if b.stamina > 0:
                    b.stamina -= 0.01*self.dt

    def update_age(self):
        for b in self.predators + self.prey:
            b.age += self.dt

    def roulette_selection(self,weights):
        cumulative = np.cumsum(weights)
        r = np.random.random() * cumulative[-1]
        for i, w in enumerate(cumulative):
            if r < w: return i

    def kill_boids(self, boids):
        # Kill boids
        number_of_killed = 0
        for b in boids:
            if (b.killed == True):
                boids.remove(b)
                number_of_killed += 1
        for i in range(number_of_killed):                
            fitness_values = [b.fitness for b in boids]
            parent = self.roulette_selection(fitness_values)
            child = type(boids[0])(self)
            child.weights = boids[parent].weights
            child.weights = child.mutate()
            boids.append(child)

    def kill_prey(self):
        self.kill_boids(self.prey)

    def kill_predators(self):
        self.kill_boids(self.predators)
        
class Boid:

    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.stamina = 1.0 # in range [0, 1] ?
        self.eating = False
        self.age = 1

    @property
    def sensors(self):
        raise NotImplementedError("Should be specialised by the subclass")

    @property
    def fitness(self):
        raise NotImplementedError("Should be specialised by the subclass")

    @property
    def acceleration(self):
        """
        Call neural network with self.sensors as input to compute acceleration

        Doesn't need to be specialised by the subclass, since the computation
          is effectively the same
        """
        position_norm = quick_norm(self.position)
        if position_norm > self.ecosystem.world_radius:
#            boundary_acc = -self.position/self.ecosystem.world_radius
            #This could be used as a "force field":
            boundary_acc = - (self.position*np.exp( # A suitable scaling parameter is needed
                1.0*(position_norm-self.ecosystem.world_radius))/position_norm)

            return self.sensors/self.boid_weight + boundary_acc
        else:
            return self.sensors/self.boid_weight # use neural work instead!

    def update_velocity(self, dt):
        self.velocity += self.acceleration * dt

    def update_position(self, dt):
        self.position += self.velocity * dt
        
    def initialize_position(self):
        angle = 2*np.pi*np.random.random()
        radius_temporary = np.random.random() + np.random.random()
        if (radius_temporary > 1):
            radius = (2-radius_temporary)*self.ecosystem.world_radius
        else:
            radius = radius_temporary*self.ecosystem.world_radius
        return np.array([radius*np.cos(angle), radius*np.sin(angle)])
        
    def initialize_velocity(self):
        angle = 2*np.pi*np.random.random()
        magnitude_temporary = np.random.random() + np.random.random()
        if (magnitude_temporary > 1):
            magnitude = (2-magnitude_temporary)*self.maximum_speed
        else:
            magnitude = magnitude_temporary*self.maximum_speed
        return np.array([magnitude*np.cos(angle), magnitude*np.sin(angle)])

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
            if (np.random.random() < self.ecosystem.mutation_probability):
                mutated_weights[i] -= 2*self.ecosystem.creep_range*(np.random.random()-0.5)
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
        return fast_boids.find_visible_neighbours(
            tree.data, self.position, self.velocity, self.perception_angle,
            self.find_neighbours(tree, radius))
        
    @property
    def killed(self):
        return False

class Prey(Boid):
    def __init__(self, ecosystem):
        Boid.__init__(self, ecosystem) # call the Boid constructor, too
        
        self.number_of_weights = 5
        self.maximum_speed = self.ecosystem.prey_maximum_speed
        self.minimum_speed = self.ecosystem.prey_minimum_speed
        self.perception_length = self.ecosystem.prey_perception_length
        self.perception_angle = self.ecosystem.prey_perception_angle
        self.too_close_radius = self.ecosystem.prey_too_close_radius
        self.boid_weight = self.ecosystem.prey_weight
        self.life_span = self.ecosystem.prey_lifespan

        self.weights = self.ecosystem.prey_network_weights # neural net weights        
        self.position = self.initialize_position()        
        self.velocity = self.initialize_velocity()

    def update_velocity(self, dt):
        """
        Checks if prey has collided with other prey. We don't have to check if
        has collided with predator since this is done in property killed.
        If colided: keep current flight direction and set speed to self.minimum_speed.
        Else: calculate new acceleration and new velocity.
        If new velocity > self.maximum_speed: Reduce speed to self.maximum_speed.
        The -1 in the len(collided_with) comes from the tree alwys returning
        the boid itself as a collision if a prey does a lookup in the prey_tree.
        """
        collided_with = self.find_neighbours(self.ecosystem.prey_tree, 2*self.ecosystem.prey_radius)
        relative_positions = np.array(
                self.ecosystem.prey_tree.data[collided_with,:] - self.position)
        relative_positions = relative_positions[np.logical_and(relative_positions[:,0]!=0,relative_positions[:,1]!=0,)]
        self.velocity += self.acceleration * dt
        velocity_norm = quick_norm(self.velocity)
        number_of_collisions = np.size(relative_positions)/2
        if number_of_collisions > 0:
            self.maximum_speed = self.minimum_speed
            if (number_of_collisions == 1):
                collision_center_of_mass = (self.position + 0.5*relative_positions.flatten())
            else:
                 collision_center_of_mass = self.position + 0.5*np.sum(relative_positions,axis=0)
            distance_to_center = quick_norm(collision_center_of_mass - self.position)
            overlap = self.ecosystem.prey_radius - distance_to_center
            collision_acc = ( - 0.01*(collision_center_of_mass - self.position)*
                np.exp(0.1*overlap)/distance_to_center)
            self.velocity += collision_acc * dt
            
        elif (self.maximum_speed < self.ecosystem.prey_maximum_speed):
            self.maximum_speed += self.ecosystem.prey_collision_speed_rebound
        velocity_norm = quick_norm(self.velocity)
        if (velocity_norm > self.maximum_speed):
            self.velocity = self.velocity*(self.maximum_speed/velocity_norm)
    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
        
        This function currently assumes prey_tree and predator_tree are of
        the class cKDTree, NOT KDTree!
        """
        sensors = np.zeros([self.number_of_weights,2])
        
        # Find visible prey.
        visible_prey_index = self.find_visible_neighbours(
            self.ecosystem.prey_tree, self.perception_length + self.ecosystem.prey_radius)
        number_of_visible_prey = len(visible_prey_index)

        if (number_of_visible_prey > 0):        
            # Calculate fellow prey position relative to self.
            relative_prey_positions = (self.ecosystem.prey_tree.data[visible_prey_index,:] - 
                self.position)
            # Calculate fellow prey velocity relative to self.
            relative_prey_velocities = (
                self.ecosystem.prey_velocities[visible_prey_index,:]-self.velocity)
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
            self.ecosystem.prey_tree, self.too_close_radius + self.ecosystem.prey_radius)
        number_of_too_close = np.size(too_close_index)
            
        if (number_of_too_close > 0):
            # Calculate "too close" sensor value.
            relative_too_close_positions = np.array(
                self.ecosystem.prey_tree.data[too_close_index,:] - self.position)
            if (number_of_too_close == 1):
                #if relative_too_close_positions.size != 2: raise Exception()
                too_close_distance = quick_norm(relative_too_close_positions.flatten())
                too_close_direction = relative_too_close_positions/too_close_distance
            else:
                too_close_distance = np.linalg.norm(relative_too_close_positions, axis=1)
                too_close_direction = (
                    relative_too_close_positions/too_close_distance[:,np.newaxis])
            sensors[2,:] = (np.dot(((self.too_close_radius/too_close_distance) - 1),
                too_close_direction)/number_of_too_close)

        # Find visible predators.
        visible_predator_index = self.find_visible_neighbours(
            self.ecosystem.predator_tree, self.perception_length + self.ecosystem.predator_radius)
        number_of_visible_predators = np.size(visible_predator_index)

        if (number_of_visible_predators > 0):
            # Calculate predator sensor.
            relative_predator_positions = np.array(
                self.ecosystem.predator_tree.data[visible_predator_index,:] - self.position)
            if (number_of_visible_predators == 1):
                distance_to_predator = quick_norm(relative_predator_positions.flatten())
                direction_to_predator = (relative_predator_positions/distance_to_predator)
            else:
                distance_to_predator = np.linalg.norm(relative_predator_positions,axis=1)
                direction_to_predator = (
                    relative_predator_positions/distance_to_predator[:,np.newaxis])
            sensors[3,:] = (np.dot(((self.perception_length/distance_to_predator) - 1),
                direction_to_predator)/number_of_visible_predators)
            
        # Feeding area sensor, assuming only one area and perfect vision.
        relative_feeding_position = (self.ecosystem.feeding_area_position-self.position)
#        sensors[4,:] = relative_feeding_position
        sensors[4,:] = np.zeros(2)

        force = np.dot(self.weights,sensors)/self.number_of_weights
        return force

    @property
    def fitness(self):
        return self.age
    
    @property
    def killed(self):
        """
        Return a boolean value describing whether boid is dead.
        Prey can die of collision with predator and old age (for now).
        """
        collided_with = self.find_neighbours(self.ecosystem.predator_tree,
            self.ecosystem.prey_radius + self.ecosystem.predator_radius)
        
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
        self.maximum_speed = self.ecosystem.predator_maximum_speed # how large?
        self.minimum_speed = self.ecosystem.predator_minimum_speed
        self.perception_length = self.ecosystem.predator_perception_length
        self.perception_angle = self.ecosystem.predator_perception_angle
        self.too_close_radius = self.ecosystem.predator_too_close_radius
        self.boid_weight = self.ecosystem.predator_weight
        self.life_span = self.ecosystem.predator_lifespan
        self.weights = self.ecosystem.predator_network_weights

        self.position = self.initialize_position()        
        self.velocity = self.initialize_velocity()
        
        self.kill_count = 1

    def update_velocity(self, dt):
        """
        Checks if prey has collided with other prey. We don't have to check if
        has collided with predator since this is done in property killed.
        If colided: keep current flight direction and set speed to self.minimum_speed.
        Else: calculate new acceleration and new velocity.
        If new velocity > self.maximum_speed: Reduce speed to self.maximum_speed.
        The -1 in len(collided_with_predator)-1 comes from the tree always 
        returning the boid itself if a predator looks up neighbours in the 
        predator_tree.
        """
        collided_with_predator = self.find_neighbours(self.ecosystem.predator_tree,
            2*self.ecosystem.predator_radius)

        collided_with_prey = self.find_neighbours(self.ecosystem.prey_tree,
              self.ecosystem.predator_radius + self.ecosystem.prey_radius)
            
        self.velocity += self.acceleration * dt
        number_of_predator_collisions = len(collided_with_predator)-1
        velocity_norm = quick_norm(self.velocity)
        if number_of_predator_collisions > 0:
            self.maximum_speed = self.minimum_speed
            relative_positions = np.array(
                self.ecosystem.predator_tree.data[collided_with_predator,:] - self.position)
            relative_positions = relative_positions[((relative_positions != np.array([0,0])))]
            if (number_of_predator_collisions == 1):
                collision_center_of_mass = self.position + 0.5*relative_positions
            else:
                 collision_center_of_mass = self.position + 0.5*np.sum(relative_positions,axis=0)
            distance_to_center = quick_norm(collision_center_of_mass - self.position)
            overlap = self.ecosystem.predator_radius - distance_to_center
            collision_acc = ( - 0.001*(collision_center_of_mass - self.position)*
                np.exp(0.1*overlap)/distance_to_center)
            self.velocity += collision_acc * dt
            
        elif len(collided_with_prey) > 0:
            self.maximum_speed = self.minimum_speed
            self.kill_count += len(collided_with_prey)
        elif(self.maximum_speed < self.ecosystem.predator_maximum_speed):
            self.maximum_speed += self.ecosystem.predator_collision_speed_rebound
        velocity_norm = quick_norm(self.velocity)
        if velocity_norm > self.maximum_speed:
            self.velocity = self.velocity*(self.maximum_speed/velocity_norm)


    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
        """
        sensors = np.zeros([self.number_of_weights,2])
        
        # Find visible prey.
        visible_prey_index = self.find_visible_neighbours(
            self.ecosystem.prey_tree, self.perception_length + self.ecosystem.prey_radius)
        number_of_visible_prey = np.size(visible_prey_index)
    
        # Target prey position sensor and target velocity. Chooses the prey 
        # that is closest to the predator.        
        if (number_of_visible_prey > 0):
            relative_prey_positions = np.array(
                self.ecosystem.prey_tree.data[visible_prey_index,:] - self.position)
            prey_distance = np.linalg.norm(relative_prey_positions, axis=1)
            target_prey_index = visible_prey_index[np.argmin(prey_distance)]
            sensors[0,:] = self.ecosystem.prey_tree.data[target_prey_index,:] - self.position
            sensors[1,:] = self.ecosystem.prey_velocities[target_prey_index,:] - self.velocity

        
        # Find visible predators.
        visible_predator_index = self.find_visible_neighbours(
            self.ecosystem.predator_tree, self.perception_length + self.ecosystem.predator_radius)
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
            self.ecosystem.predator_tree, self.too_close_radius + self.ecosystem.predator_radius)
        number_of_too_close = np.size(too_close_index)
        
        # Calculate "too close" sensor value.
        if (number_of_too_close > 0):
            relative_too_close_positions = np.array(
                self.ecosystem.predator_tree.data[too_close_index,:] - self.position)
            if (number_of_too_close == 1):
                #if relative_too_close_positions.size != 2: raise Exception()
                too_close_dist = quick_norm(relative_too_close_positions.flatten())
                too_close_direction = (relative_too_close_positions/too_close_dist)
            else:
                too_close_dist = np.linalg.norm(relative_too_close_positions, axis=1)
                too_close_direction = (
                    relative_too_close_positions/too_close_dist[:,np.newaxis])
            sensors[4,:] = (np.dot(((self.too_close_radius/too_close_dist)-1),
                too_close_direction)/number_of_visible_predators)
    
        # Total force.           
        force = np.dot(self.weights,sensors)/self.number_of_weights
        return force
        
    @property
    def fitness(self):
        return self.kill_count

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
