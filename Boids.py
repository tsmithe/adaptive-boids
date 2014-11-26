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
- Mutate network (for offspring)
- 

Property functions
- Compute sensor values, 2xn array
- Update position
- Update velocity
- Update acceleration
- Update stamina
- Update killed (Check if living)
- 

"""
    def __init__(self, worldsize):
        self.position = np.random.random(2)*worldsize