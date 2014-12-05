import csv
import numpy as np

class StatisticsHelper:
    
    def __init__(self, file_path, append_to_file = False,
                 position = False,
                 velocities = False,
                 centre_of_mass = False,
                 scalars = False,
                 ):
        '''
        The named parameters correspond to statistical numbers that can be
        exported. They will only be exported if the parameter is set to true.
        '''
        if append_to_file: self.file_mode = 'a'
        else: self.file_mode = 'w'
        
        if position:
            self.positions_csv = csv.writer(open(file_path+"positions.csv",
                                                 self.file_mode))
        else:
            self.positions_csv = None
            
        if velocities:
            self.velocities_csv = csv.writer(open(file_path+"velocities.csv",
                                                  self.file_mode))
        else:
            self.velocities_csv = None

        if centre_of_mass:
            self.centre_of_mass_csv = csv.writer(open(
                file_path+"centre_of_mass.csv",
                self.file_mode))
        else:
            self.centre_of_mass_csv = None
            
        if scalars:
            self.scalars_csv = csv.writer(open(file_path+"scalars.csv",
                                               self.file_mode))
        else:
            self.scalars_csv = None

    def update_data(self, boids, tree):
        self.boids = boids
        self.tree = tree

    def export(self):
        if self.positions_csv:
            self.positions_csv.writerow(self.positions.tolist())
            
        if self.velocities_csv:
            self.velocities_csv.writerow(self.velocities.tolist())

        if self.centre_of_mass_csv:
            self.centre_of_mass_csv.writerow(self.centre_of_mass.tolist())
            
        if self.scalars_csv:
            self.scalars_csv.writerow([self.average_nearest_neighbour,
                                       self.average_distance_centre_of_mass])

    @property
    def positions(self):
        return np.ndarray.flatten(np.array([b.position for b in self.boids]))
    
    @property
    def velocities(self):
        return np.ndarray.flatten(np.array([b.velocity for b in self.boids]))
    
    @property
    def average_nearest_neighbour(self):
        # Closest neighbour
        distances, indices = self.tree.query([b.position for b in self.boids])
        return np.mean(distances)
    
    @property
    def centre_of_mass(self):
        # Computes the centre of mass
        position_sum = np.array([0,0])
        for b in self.boids:
            position_sum += b.position
            
        return position_sum/len(self.boids)
    
    @property
    def average_distance_centre_of_mass(self):
        # Computes the average distance to the centre of mass
        centre_of_mass = self.centre_of_mass
        distance_sum = 0;
        for b in self.boids:
            distance_sum += np.linalg.norm(b.position-centre_of_mass)
            
        return distance_sum/len(self.boids)
