# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 09:18:25 2014

@author: rikardvinge
"""
from numpy import *
import scipy
import scipy.linalg
import random

class Boid: # Temporary Boid class
    def __init__(self, initialPosition, initialVelocity, networkWeights, boidWeight, dt, maxVelocity):
        self.position = initialPosition # should be size (1,2)
        self.velocity = initialVelocity # should be size (1,2)
        self.networkWeights = networkWeights # should be size (n,1)
        self.numberOfNetworkWeights = size(self.networkWeights)
        self.weight = boidWeight
        self.dt = dt
        self.maxVelocity = maxVelocity
        
    def GetAcceleration(self):
        sensorInformation = self.GetSensorInformation()
        weightedSensorInformation = zeros((self.numberOfNetworkWeights,2))
        weightedSensorInformation[:,0] = self.networkWeights*sensorInformation[:,0].copy()
        weightedSensorInformation[:,1] = self.networkWeights*sensorInformation[:,1].copy()
        acceleration = sum(weightedSensorInformation/self.weight,0)
        return acceleration

    def GetSensorInformation(self):
        # Add code for different sensors
        # Currently only returning a [n,2] array of ones
        sensorInformation = ones((self.numberOfNetworkWeights,2))
        return sensorInformation
        
    def UpdateVelocity(self):
        acceleration = self.GetAcceleration()
        velocity = self.velocity + self.dt*acceleration
        speed = linalg.norm(velocity)
        if speed > self.maxVelocity:
            self.velocity = velocity.copy()*self.maxVelocity/speed
        else:
            self.velocity = velocity.copy()
        return
        
    def UpdatePosition(self):
        self.UpdateVelocity()
        self.position = self.position + self.dt*self.velocity
        return
        
        
class Environment: # Temporary class for background environment
    def __init__(self, areaRadius):
        self.radius = areaRadius
        
    