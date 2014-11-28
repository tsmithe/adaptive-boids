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
    def __init__(self, worldsize):
        self.position = np.random.random(2)*worldsize
        self.velocity = np.random.random(2) # TODO: decide range
        self.stamina = 1.0 # in range [0, 1] ?
        self.eating = False
        self.age = 0
        self.creeprange = 0.1 # how large?
        self.maximumspeed = 1 # how large?
        self.viewangle = np.pi/2 # how large? Should it differ between prey/predators

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
        """
        Update velocity by calling acceleration property.
        If speed is higher than maximumspeed, then decrease speed to 
        maximumspeed but keep direction.
        """
        self.velocity += self.acceleration * dt
        currentspeed = np.sqrt(np.dot(self.velocity,self.velocity))
        if (currentspeed > self.maximumspeed):
            self.velocity *= self.maximumspeed/currentspeed

    def update_position(self, dt):
        self.update_velocity(dt)
        self.position += self.velocity * dt

    def mutate(self):
        """
        Mutate neural network weights.
        Implemented as a linearly distributed creep mutation.
        Returns a network with weights mutated with linear creep within
        [-creeprange,creeprange] from the original weights.
        No upper or lower bounds.
        """
        network_size = np.size(self.weights)
        mutated_weights = (self.weights.copy() - 
            2*self.creeprange*(np.random.random(network_size)-0.5))
        return mutated_weights
        
    def visible_neighbours(self, neighbourpositionarray):
        """
        Takes array of nearest-neighbour position as input.
        Checks if each nearest-neighbour is within the viewangle of the boid.
        Returns indexes, of the neighbourpositionarray, that the boid can see.
        
        TODO: 
        - Include if statement that checks if angle is less than 
        viewingangle.
        - Create 1xm array that will contain the indices of the neighbours within
        sight. How do we create this array when we don't know how many 
        visible neighbours there will be?
        - Fill the index array and return
        """
        numberofneighbours = np.size(neighbourpositionarray)/2;
        for i in np.arange(numberofneighbours):
            relativeposition = neighbourpositionarray[0,:] - self.position
            neighbourdistance = np.sqrt(np.dot(relativeposition,relativeposition))
            currentspeed = np.sqrt(np.dot(self.velocity,self.velocity))
            angle = np.arccos(np.dot(relativeposition,self.velocity)/
                (neighbourdistance*currentspeed))
        return
        

    @property
    def killed(self):
        return False

class Prey(Boid):
    def __init__(self, worldsize):
        Boid.__init__(self, worldsize) # call the Boid constructor, too

        self.numberofweights = 5
        self.weights = np.random.random(self.numberofweights) # neural net weights

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2

        Do something different from Prey.sensors!
        """
        return np.random.random(
            2*self.numberofweights).reshape(self.numberofweights,2)

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
    def __init__(self, worldsize):
        Boid.__init__(self, worldsize) # call the Boid constructor, too

        self.numberofweights = 6
        self.weights = np.random.random(self.numberofweights) # neural net weights

    @property
    def sensors(self):
        """
        Compute input sensors values to neural network
        Returns an array of sensor values of shape n x 2

        Do something different from Prey.sensors!
        """
        return np.random.random(
            2*self.numberofweights).reshape(self.numberofweights,2)
