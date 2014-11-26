# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:00:03 2014

@author: rikardvinge
"""

import numpy as np

class Boid:
"""
Variables
- Position, 1x2 array
- Velocity, 1x2 array
- Acceleration, 1x2 array
- Network weights, 1xn array
- Collided, boolean
- Stamina, double
- Eating, boolean
- Killed, boolean
- Region/tree?
- Age, integer
- Maximum speed, double

Class functions
- Update position
- Update velocity
- Mutate network (for offspring)
- 

Property functions
- Compute sensor values, 2xn array
- Update acceleration
- Update killed (Check if living)
- 

"""
    def __init__(self, worldsize):
        self.position = np.random.random(2)*worldsize
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
    def __init__(self, worldsize):
        Boid.__init__(self, worldsize) # call the Boid constructor, too

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
        return np.random.random(10).reshape(6,2)
