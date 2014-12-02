import numpy as np

class environment:
    def __init__(self, radius):
        self.radius = radius
        
    def is_feeding(self, boid):
        if np.linalg.norm(boid.position) < self.radius:
            return True
        else:
            return False