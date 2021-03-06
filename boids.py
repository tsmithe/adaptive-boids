# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

"""

import numpy as np
from scipy.spatial import cKDTree
import random

import fast_boids
from fast_boids import quick_norm, quick_dot

class Ecosystem:
    def __init__(self, config):

        self.feeding_areas = FeedingAreas()
        area_helper = FeedingAreaConfigurations()
        area_helper.initialize_feeding_areas(eval(config['DEFAULT']['feeding_areas']), self)

        self.dt = eval(config['DEFAULT']['dt'])
        self.world_radius = eval(config['DEFAULT']['world_radius'])
        self.num_prey = eval(config['DEFAULT']['num_prey'])
        self.num_predators = eval(config['DEFAULT']['num_predators'])
        self.weights_distribution_std = eval(config['DEFAULT']['weights_distribution_std'])

        self.prey_radius = eval(config['Prey']['boid_radius'])
        self.prey_max_speed = eval(config['Prey']['boid_max_speed'])
        self.prey_min_speed = eval(config['Prey']['boid_min_speed'])
        self.prey_collision_speed_rebound = ((self.prey_max_speed-self.prey_min_speed)*
                                             eval(config['Prey']['boid_collision_recovery_rate']))
        self.prey_max_steering_angle = eval(config['Prey']['boid_max_steering_angle'])
        self.prey_max_force = eval(config['Prey']['boid_max_force'])
        self.prey_perception_length = eval(config['Prey']['boid_perception_length'])
        self.prey_perception_angle = eval(config['Prey']['boid_perception_angle'])
        self.prey_too_close_radius = eval(config['Prey']['boid_too_close_radius'])
        self.prey_weight = eval(config['Prey']['boid_weight'])
        self.prey_lifespan = eval(config['Prey']['boid_lifespan'])
        try:
            self.prey_lifespan_increase_rate = eval(config['Prey']['boid_lifespan_increase_rate'])
        except KeyError:
            self.prey_lifespan_increase_rate = 1.0

        self.predator_radius = eval(config['Predator']['boid_radius'])
        self.predator_max_speed = eval(config['Predator']['boid_max_speed'])
        self.predator_min_speed = eval(config['Predator']['boid_min_speed'])
        self.predator_collision_speed_rebound = ((self.predator_max_speed-self.predator_min_speed)*
                                             eval(config['Predator']['boid_collision_recovery_rate']))
        self.predator_max_steering_angle = eval(config['Predator']['boid_max_steering_angle'])
        self.predator_max_force = eval(config['Predator']['boid_max_force'])
        self.predator_perception_length = eval(config['Predator']['boid_perception_length'])
        self.predator_perception_angle = eval(config['Predator']['boid_perception_angle'])
        self.predator_too_close_radius = eval(config['Predator']['boid_too_close_radius'])
        self.predator_weight = eval(config['Predator']['boid_weight'])
        self.predator_lifespan = eval(config['Predator']['boid_lifespan'])
        try:
            self.predator_lifespan_increase_rate = eval(config['Predator']['boid_lifespan_increase_rate'])
        except KeyError:
            self.predator_lifespan_increase_rate = 1.0

        self.prey = []
        self.predators = []
        
        self.best_predator_fitness = 0
        self.best_predator_weights = None

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
        
    def update_age(self):
        for b in self.predators + self.prey:
            b.is_feeding()
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
            
            if self.feeding_areas.number_of_areas() > 0:
                child.position = self.feeding_areas.get_random_position()
            
            boids.append(child)

    def kill_prey(self):
        self.kill_boids(self.prey)

    def kill_predators(self):
        # kill predators.
        # Predator with highest kill_count is the parent of all new predators.
        fitness_values = [b.fitness for b in self.predators]
        max_fitness_predator_index = np.argmax(fitness_values)
        if (fitness_values[max_fitness_predator_index] > self.best_predator_fitness
        or self.best_predator_weights is None):
            self.best_predator_fitness = fitness_values[max_fitness_predator_index]
            self.best_predator_weights = self.predators[max_fitness_predator_index].weights

        for p in self.predators:
            if (p.killed == True):
                self.predators.remove(p)
                child = Predator(self)
                child.weights = self.best_predator_weights
                child.weights = child.mutate()
                self.predators.append(child)


class FeedingAreas:
    '''
    Each cKDTree can only handle one radius for all feeding areas in it.
    Therefore a larger structure is necessary. This class collects
    FeedingAreaGroup. Each FeedingAreaGroup collects feeding areas with the
    same radius.
    '''
    
    def __init__(self):
        self.areas = []
        
    def add_feeding_area(self,feeding_area):
        self.areas.append(feeding_area)
        
    def is_feeding(self,boid):
        for a in self.areas:
            if a.is_feeding(boid.position):
                return True
        return False
    
    def closest_feeding_area(self, boid):
        closest_distance = np.inf
        closest_area_location = None
        closest_area_radius = None
        for a in self.areas:
            selected_position = a.closest_feeding_area_position(boid)
            distance = quick_norm(boid.position-selected_position)
            if distance < closest_distance:
                closest_distance = distance
                closest_area_location = selected_position
                closest_area_radius = a.radius
                
        return (closest_area_location, closest_area_radius)
    
    def number_of_areas(self):
        return len(self.areas)
    
    def roulette_selection(self,weights):
        cumulative = np.cumsum(weights)
        r = np.random.random() * cumulative[-1]
        for i, w in enumerate(cumulative):
            if r < w: return i
            
    def get_random_position(self):
        """
        Selects an feeding area with roulettewheel selection proportional 
        to its area. Uses the same algorithm as intialize_position() to chose
        a random point inside the chosen feeding area.
        """
        weights = []
        for area in self.areas:
            weights.append(area.radius**2)
        selected_index = self.roulette_selection(weights)
        selected_area = self.areas[selected_index]

        radius_temporary = np.random.random() + np.random.random()
        if (radius_temporary > 1):
            radius = (2-radius_temporary)*selected_area.radius
        else:
            radius = radius_temporary*selected_area.radius

        theta = random.random()*2*np.pi
        
        # Recall that there is an intermediate layer, the feeding group,
        # which may encapsulate several feeding areas of the same radius.
        position = random.choice(selected_area.positions)
        
        return np.array([position[0] + radius*np.cos(theta), position[1] + radius*np.sin(theta)])
    
class FeedingAreaGroup:
    '''
    Represents a group of feeding areas, all with
    the same radius.
    '''
    
    def __init__(self, positions, radius):
        self.radius = radius
        self.positions = positions
        self.tree = cKDTree(positions)
        
    def is_feeding(self, pt):
        hits = self.tree.query_ball_point(pt,self.radius)
        if len(hits) > 0:
            return True
        return False
    
    def closest_feeding_area_index(self, boid):
        distance, index = self.tree.query(boid.position)
        return index

    def closest_feeding_area_position(self, boid):
        feeding_area_index = self.closest_feeding_area_index(boid)
        return self.tree.data[feeding_area_index,:]


class FeedingAreaConfigurations:
    '''
    Note that even though the system supports several different groups
    of feeding areas with different radii, there is currently no way to
    convey such information to the visualization.
    '''
    configurations = {
        'centered_feeding_area': ([[0,0]], 500),
        'twins': ([[-50,0],[50,0]], 25),
        'triplets': ([[-50, -50], [50, 0], [-50, 50]], 20)
    }
                      
    def initialize_feeding_areas(self, name_or_config, ecosystem):
        if isinstance(name_or_config, tuple):
            config = name_or_config
            feeding_area_locations = np.array(config[0])
            feeding_area_group = FeedingAreaGroup(feeding_area_locations,
                                                  config[1])
            ecosystem.feeding_areas.add_feeding_area(feeding_area_group)
            return

        else:
            name = name_or_config
        if name in self.configurations.keys():
            feeding_area_locations = np.array(self.configurations[name][0])
            feeding_area_group = FeedingAreaGroup(feeding_area_locations,
                                                  self.configurations[name][1])
            ecosystem.feeding_areas.add_feeding_area(feeding_area_group)
    
    def get_info(self, name):
        if name in self.configurations.keys():
            return self.configurations[name]
        else:
            return ([],[])


class Boid:
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
    5. Perimeter sensor
    6. Feeding area sensor
    '''

    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.stamina = 1.0 # in range [0, 1] ?
        self.eating = False
        self.collision_overlap_sum = 0.0
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
        # Get force from sensors.
        return self.sensors/self.boid_weight
        
    def limit_direction_change(self, new_velocity):
        # Limit change in direction
        new_velocity_norm = quick_norm(new_velocity)
        current_velocity_norm = quick_norm(self.velocity)
        
        # Find angle between wanted velocity and current velocity
        cosine = quick_dot(new_velocity,self.velocity)/(new_velocity_norm*current_velocity_norm)
        cosine = np.clip(cosine, 0., 1.)
        angle = np.arccos(cosine)
        delta_angle = angle - self.max_steering_angle
        # Test if boid tries to turn too much.
        if(delta_angle > 0):
            # Boid wants to turn too much: limit turning to max_steering_angle while
            # keeping turning direction and magnitude of new velocity.
        
            # Find vector 90 degrees counter-clockwise from current direction
            perpendicular_to_current_velocity = np.array([-self.velocity[1], self.velocity[0]])
            if (quick_dot(new_velocity,perpendicular_to_current_velocity) > 0):
                # new_velocity is "to the left" of self.velocity.
                # Rotate new_velocity delta_angle clockwise.
                delta_angle = -delta_angle
                # Otherwise, rotate counter-clockwise, i.e. delta-angle should be > 0
            
            # Create rotation matrix
            rotation_matrix = np.array([[np.cos(delta_angle), -np.sin(delta_angle)],
                                         [np.sin(delta_angle), np.cos(delta_angle)],])
            # Rotate new_velocity without changing magnitude.
            new_velocity = np.dot(rotation_matrix,new_velocity)
        return new_velocity

    def collision_check(self, new_velocity, tree, self_radius, collision_radius):
        # Check if boid collided with other boids in tree.
        collided_with = self.find_neighbours(tree, collision_radius)
        # Calculate relative position vectors from self to other boids in collision,
        # the boid itself is in this array.
        relative_positions = np.array(tree.data[collided_with,:] - self.position)
        # Remove the boid iteslf from array.
        relative_positions = relative_positions[np.logical_and(
            relative_positions[:,0]!=0,relative_positions[:,1]!=0)]
            
        # Check if boid collided with other boids.
        number_of_collisions = np.size(relative_positions)/2
        if number_of_collisions > 0:
            # Reduce maximum velocity
            self.max_speed = self.min_speed
 
            if (number_of_collisions == 1):
                # Collided with one other predator.
                # Calculate distance to boids in collision.
                distance_between_boids = quick_norm(relative_positions.flatten())
                # Calculate overlap.
                overlap = 2*self_radius - distance_between_boids
                # Increase collision_overlap_sum.
                self.collision_overlap_sum += overlap/2
                # Relative position unit vector.
                relative_unit_vector = relative_positions/distance_between_boids

                # Calculate collision acceleration (Basically Pauli exclusion).
                collision_acc = -(relative_unit_vector)*np.exp(overlap)
            else:
                # Collided with multiple predators.
                # Calculate distance to boids in collision.
                distance_between_boids = np.linalg.norm(relative_positions, axis=1)
                # Calculate boid overlap.
                overlap = 2*self_radius - distance_between_boids
                # Increase the collision_overlap_sum with all the overlaps
                self.collision_overlap_sum += np.sum(overlap)/2
                # Relative position unit vector.
                relative_unit_vector = relative_positions/distance_between_boids[:,np.newaxis]
                # Calculate collision acceleration (Basically Pauli exclusion).
                collision_acc = np.sum(-(relative_unit_vector)*np.exp(overlap[:,np.newaxis]),axis=0)/number_of_collisions
                
            collision_acc = collision_acc.flatten()
            new_velocity += collision_acc * self.ecosystem.dt

        # Return new_velocity   
        return new_velocity

    def update_velocity(self, dt):
        self.velocity += self.acceleration * dt

    def update_position(self, dt):
        self.position += self.velocity * dt
        
    def initialize_position(self):
        if self.ecosystem.feeding_areas.areas:
            return self.ecosystem.feeding_areas.get_random_position()

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
            magnitude = (2-magnitude_temporary)*self.max_speed
        else:
            magnitude = magnitude_temporary*self.max_speed
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
        mutated_weights = self.weights + np.random.normal(0.0,
            self.ecosystem.weights_distribution_std, self.number_of_weights)
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
        
        self.number_of_weights = 6
        self.pick_weights_sd = self.ecosystem.weights_distribution_std
        self.max_speed = self.ecosystem.prey_max_speed
        self.original_max_speed = self.ecosystem.prey_max_speed
        self.min_speed = self.ecosystem.prey_min_speed
        self.max_steering_angle = self.ecosystem.prey_max_steering_angle
        self.perception_length = self.ecosystem.prey_perception_length
        self.perception_angle = self.ecosystem.prey_perception_angle
        self.too_close_radius = self.ecosystem.prey_too_close_radius
        self.boid_weight = self.ecosystem.prey_weight
        self.lifespan = self.ecosystem.prey_lifespan
        self.collision_rebound_rate = self.ecosystem.prey_collision_speed_rebound

        self.weights = np.random.normal(0.0,self.pick_weights_sd,self.number_of_weights)
        self.position = self.initialize_position()        
        self.velocity = self.initialize_velocity()

    def update_velocity(self, dt):
        """
        Checks if prey has collided with other prey. We don't have to check if
        has collided with predator since this is done in property killed.
        If colided: keep current flight direction and set speed to self.min_speed.
        Else: calculate new acceleration and new velocity.
        If new velocity > self.max_speed: Reduce speed to self.max_speed.
        The -1 in the len(collided_with) comes from the tree alwys returning
        the boid itself as a collision if a prey does a lookup in the prey_tree.
        """
        # Increase maximum speed in case it is lower than its original value.
        if(self.max_speed < self.original_max_speed):
            self.max_speed += self.collision_rebound_rate
            
        # Get acceleration from sensors and calculate wanted new velocity.
        new_velocity = self.velocity + self.acceleration * dt
        
        # Check for collisions and limit max speed and reduce current speed in case of collison.
        new_velocity = self.collision_check(new_velocity, self.ecosystem.prey_tree,  
            self.ecosystem.prey_radius, 2*self.ecosystem.prey_radius)
            
        # Limit change of direction.
        new_velocity = self.limit_direction_change(new_velocity)
        velocity_norm = quick_norm(new_velocity)
        
        # Check if boid is outside permieter. If so, limit speed.
        if (quick_norm(self.position) > self.ecosystem.world_radius):
            self.max_speed = self.min_speed
        
        # Limit speed to self.max_speed.
        if (velocity_norm > self.max_speed):
            new_velocity = new_velocity*(self.max_speed/velocity_norm)
        
        # Update velocity.
        self.velocity = new_velocity
        
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
        
        # Perimeter sensor
        # Prey distance to origin.
        radial_position = quick_norm(self.position)
        # Perimeter visible to prey?
        distance_to_boundary = self.ecosystem.world_radius - radial_position
        if (distance_to_boundary < self.perception_length):
            # Perimeter visible.
            sensors[4,:] = (((self.perception_length/distance_to_boundary) - 1)*
                self.position/radial_position)
            
        # Feeding area sensor
        if self.ecosystem.feeding_areas.number_of_areas() > 0:
            (feeding_area_position, feeding_area_radius) = self.ecosystem.feeding_areas.closest_feeding_area(self)
            distance_to_center = quick_norm(feeding_area_position-self.position)
            if distance_to_center-feeding_area_radius > 0:
                sensors[5,:] = ((distance_to_center-feeding_area_radius)*
                    (feeding_area_position-self.position)/
                    (distance_to_center*(self.lifespan-self.age+1)))
            else:
                sensors[5,:] = 0

        force = np.dot(self.weights,sensors)/self.number_of_weights
        force_norm = quick_norm(force)
        if (force_norm > self.ecosystem.prey_max_force):
            force *= self.ecosystem.prey_max_force/force_norm
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
            
        # Check if outise perimeter
        radial_position = quick_norm(self.position)
        how_far_out = (radial_position + self.ecosystem.predator_radius - 
            self.ecosystem.world_radius)

        if len(collided_with) > 0:
            # Kill if eaten by predator
            return True
        elif (how_far_out > 0.5):
            # Kill if too far outside. (0.5 is apparently far enough according to Sayers)
            return True
        elif (self.collision_overlap_sum >= self.ecosystem.prey_radius):
            # Kill if the prey has collided too much/many times
            return True
        elif (self.age > self.lifespan):
            # Kill if too old.
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

    def is_feeding(self):
        if self.ecosystem.feeding_areas.is_feeding(self):
            self.lifespan += self.ecosystem.prey_lifespan_increase_rate*self.ecosystem.dt
        
        
class Predator(Boid):
    def __init__(self, ecosystem):
        Boid.__init__(self, ecosystem) # call the Boid constructor, too

        self.number_of_weights = 7
        self.pick_weights_sd = self.ecosystem.weights_distribution_std
        self.max_speed = self.ecosystem.predator_max_speed
        self.original_max_speed = self.ecosystem.predator_max_speed
        self.min_speed = self.ecosystem.predator_min_speed
        self.max_steering_angle = self.ecosystem.predator_max_steering_angle
        self.perception_length = self.ecosystem.predator_perception_length
        self.perception_angle = self.ecosystem.predator_perception_angle
        self.too_close_radius = self.ecosystem.predator_too_close_radius
        self.boid_weight = self.ecosystem.predator_weight
        self.lifespan = self.ecosystem.predator_lifespan
        self.collision_rebound_rate = self.ecosystem.predator_collision_speed_rebound
        
        self.weights = np.random.normal(0.0,self.pick_weights_sd,self.number_of_weights)
        self.position = self.initialize_position()        
        self.velocity = self.initialize_velocity()
        
        self.kill_count = 0

    def update_velocity(self, dt):
        """
        Checks if prey has collided with other prey. We don't have to check if
        has collided with predator since this is done in property killed.
        If colided: keep current flight direction and set speed to self.min_speed.
        Else: calculate new acceleration and new velocity.
        If new velocity > self.max_speed: Reduce speed to self.max_speed.
        The -1 in len(collided_with_predator)-1 comes from the tree always 
        returning the boid itself if a predator looks up neighbours in the 
        predator_tree.
        """
        
        # Increase maximum speed in case it is lower than its original value.
        if(self.max_speed < self.original_max_speed):
            self.max_speed += self.collision_rebound_rate

        # Get acceleration from sensors
        new_velocity = self.velocity + self.acceleration * dt        

        # Check if predator collided with prey, i.e. eating it.
        collided_with_prey = self.find_neighbours(self.ecosystem.prey_tree,
              self.ecosystem.predator_radius + self.ecosystem.prey_radius)
        number_of_eaten_prey = len(collided_with_prey)
        if number_of_eaten_prey > 0:
            # Slow down.
            self.max_speed = self.min_speed
            # Increase kill count.
            self.kill_count += number_of_eaten_prey
            # Increase lifespan.
            self.lifespan += number_of_eaten_prey*self.ecosystem.predator_lifespan_increase_rate*self.ecosystem.dt
            
        # Check for collisions and limit max speed and reduce current speed in case of collison.
        new_velocity = self.collision_check(new_velocity, self.ecosystem.predator_tree,  
            self.ecosystem.predator_radius, 2*self.ecosystem.predator_radius)
            
        # Limit change of direction.
        new_velocity = self.limit_direction_change(new_velocity)
        velocity_norm = quick_norm(new_velocity)
        
        # Check if boid is outside permieter. If so, limit speed.
        if (quick_norm(self.position) > self.ecosystem.world_radius):
            self.max_speed = self.min_speed
        
        # Limit velocity to self.max_speed.
        if velocity_norm > self.max_speed:
            new_velocity = new_velocity*(self.max_speed/velocity_norm)
        
        # Update velocity
        self.velocity = new_velocity


    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
        """
        sensors = np.zeros([self.number_of_weights,2])
        
        # Target prey position sensor and target velocity. Chooses the prey 
        # that is closest to the predator.
        # If predators have limited vision and view angle: use find_visible_neghbours.
        # If predators have unlimited vision: use direct targeting of prey 
        """        
        if ((self.perception_angle < np.pi) | (self.perception_length < 2*self.ecosystem.world_radius)):        
            
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
        else:
        """
        # Find closest prey.
        (target_prey_distance, target_prey_index) = (
            self.ecosystem.prey_tree.query(self.position))
        # Calculate relative position and velocity.
        relative_position = self.ecosystem.prey_tree.data[target_prey_index,:] - self.position
        relative_velocity = self.ecosystem.prey_velocities[target_prey_index,:] - self.velocity
        sensors[0,:] = relative_position
        sensors[1,:] = relative_velocity
        sensors[2,:] = relative_position + relative_velocity * self.ecosystem.dt

        # Fellow predator sensors.
        if ((self.perception_angle < np.pi) | (self.perception_length < 2*self.ecosystem.world_radius)):            
            # Find visible predators.
            visible_predator_index = self.find_visible_neighbours(
                self.ecosystem.predator_tree, self.perception_length + self.ecosystem.predator_radius)
            # Number of visible predators
            number_of_visible_predators = np.size(visible_predator_index)            
                # Fellow predator position and velocity sensor values.
            if (number_of_visible_predators > 0):
                # Relative positions.
                relative_predator_positions = np.array(
                    self.ecosystem.predator_tree.data[visible_predator_index,:]-self.position)
                # Relative velocities.
                relative_predator_velocities = np.array(
                    self.ecosystem.predator_velocities[visible_predator_index,:]-self.velocity)
                if (number_of_visible_predators == 1):
                    sensors[3,:] = relative_predator_positions
                    sensors[4,:] = relative_predator_velocities
                else:
                    sensors[3,:] = (np.sum(relative_predator_positions,axis=0)/
                        number_of_visible_predators)
                    sensors[4,:] = (np.sum(relative_predator_velocities,axis=0)/
                        number_of_visible_predators)
        else:
            fellow_predator_indices = (
                self.ecosystem.predator_tree.query_ball_point(
                self.position, 3.0*self.ecosystem.world_radius))
            relative_predator_positions = (
                np.array(self.ecosystem.predator_tree.data[fellow_predator_indices,:] -
                self.position))
            relative_predator_velocities = (
                np.array(self.ecosystem.predator_velocities[fellow_predator_indices,:] - 
                self.velocity))
            sensors[3,:] = ((np.sum(relative_predator_positions,axis=0)/
                self.ecosystem.num_predators))
            sensors[4,:] = ((np.sum(relative_predator_velocities,axis=0)/
                self.ecosystem.num_predators))

                    
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
            sensors[5,:] = (np.dot(((self.too_close_radius/too_close_dist)-1),
                too_close_direction)/number_of_too_close)
                
        # Perimeter sensor
        # Predator distance to origin.
        radial_position = quick_norm(self.position)
        # Perimeter visible to predator?
        distance_to_boundary = self.ecosystem.world_radius-radial_position
        if (distance_to_boundary < self.perception_length):
            # Perimeter visible.
            sensors[6,:] = (((self.perception_length/distance_to_boundary) - 1)*
                self.position/radial_position)
    
        # Total force.
        force = np.dot(self.weights,sensors)/self.number_of_weights
        force_norm = quick_norm(force)
        if (force_norm > self.ecosystem.predator_max_force):
            force *= self.ecosystem.predator_max_force/force_norm
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
        # Check if outise perimeter
        radial_position = quick_norm(self.position)
        how_far_out = (radial_position + self.ecosystem.predator_radius - 
            self.ecosystem.world_radius)
        # If too far out, kill boid
        if (how_far_out > 0.5):
            return True
        elif (self.age > self.lifespan):
            return True
        elif (self.collision_overlap_sum >= self.ecosystem.predator_radius):
            # Kill if the predator has collided too much/many times
            return True
        else:
            return False
            
    def is_feeding(self):
        pass
