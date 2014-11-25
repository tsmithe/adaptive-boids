# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:22:40 2014

@author: rikardvinge
"""

from numpy import *
#import scipy
#import random
import timeStepping

environmentRadius = 1
position = np.array([0.5, 0.5])
velocity = np.array([0, 1])
networkWeights = linspace(0.1, 0.6, 6)
boidWeight = 1
dt = 1
maxVelocity = 10

boid = timeStepping.Boid(position, velocity, networkWeights, boidWeight, dt, maxVelocity)
print(boid.position)
acceleration = boid.GetAcceleration()
print(acceleration)
#print(boid.velocity)
#boid.UpdateVelocity()
#print(boid.velocity)
print(boid.position)
boid.UpdatePosition()
print(boid.position)
