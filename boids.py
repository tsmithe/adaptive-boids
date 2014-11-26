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

    @property
    def killed(self):
        return False

class Prey(Boid):
    def __init__(self, worldsize):
        Boid.__init__(self, worldsize) # call the Boid constructor, too

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2
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
