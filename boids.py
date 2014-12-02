# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

@author: rikardvinge
"""

import numpy as np
from scipy.spatial import cKDTree

class Ecosystem:
    def __init__(self, world_size, num_prey, num_predators,
                 feeding_area_position, dt):
        self.dt = dt
        self.world_size = world_size
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.feeding_area_position = np.asarray(feeding_area_position)

        self.prey = []
        self.predators = []

        for i in range(self.num_prey):
            self.prey.append(Prey(self))

        for i in range(self.num_predators):
            self.predators.append(Predator(self))

        self.update_velocities()
        self.update_positions()

    def update_positions(self):
        for b in self.predators + self.prey:
            b.update_position(self.dt)
        self.prey_positions = np.array([p.position for p in self.prey])
        self.predator_positions = np.array([p.position for p in self.predators])
        #self.all_positions = np.concatenate(prey_positions, predator_positions)
        self.prey_tree = cKDTree(self.prey_positions)
        self.predator_tree = cKDTree(self.predator_positions)

    def update_velocities(self):
        for b in self.predators + self.prey:
            b.update_velocity(self.dt)
        self.prey_velocities = np.array([p.velocity for p in self.prey])
        self.predator_velocities = np.array([p.velocity for p in self.predators])


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
    - 

    Property functions
    - sensors, nx2 array
    - acceleration, 1x2 array
    - killed (Check if living), boolean
    - 

    """
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.position = np.random.random(2)*ecosystem.world_size
        self.velocity = np.random.random(2) # TODO: decide range
        self.stamina = 1.0 # in range [0, 1] ?
        self.eating = False
        self.age = 0

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
        self.velocity += self.acceleration * dt

    def update_position(self, dt):
        self.update_velocity(dt)
        self.position += self.velocity * dt

    def mutate(self):
        """
        mutate neural network weights
        """
        return

    @property
    def killed(self):
        return False

class Prey(Boid):
    def __init__(self, ecosystem):
        Boid.__init__(self, ecosystem) # call the Boid constructor, too

        self.weights = np.random.random(5) # neural net weights

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2

        Do something different from Prey.sensors!
        """
        return np.random.random(10).reshape(5,2)

    @property
    def killed(self):
        """
        return a boolean value describing whether boid is dead
        """
        death_probability = 1 - self.stamina
        if np.random.random() < death_probability:
            return False
        else:
            return True

class Predator(Boid):
    def __init__(self, worldsize):
        Boid.__init__(self, worldsize) # call the Boid constructor, too

        self.weights = np.random.random(6) # neural net weights

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2

        Do something different from Prey.sensors!
        """
        return np.random.random(12).reshape(6,2)
