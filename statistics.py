from savecsv import savecsv
import numpy as np

class StatisticsHelper:
    
    def __init__(self, boids, tree, file_path,
                 position = False,
                 direction = False,
                 centre_of_mass = False,
                 scalars = False,
                 ):
        '''
        The named parameters correspond to statistical numbers that can be
        exported. They will only be exported if the parameter is set to true.
        '''
        self.boids = boids
        self.tree = tree
        
        if position:
            self.positions_csv = savecsv(file_path+"positions.csv")
        else:
            self.positions_csv = None
            
        if direction:
            self.directions_csv = savecsv(file_path+"directions.csv")
        else:
            self.directions_csv = None

        if centre_of_mass:
            self.centre_of_mass_csv = savecsv(file_path+"centre_of_mass.csv")
        else:
            self.centre_of_mass_csv = None
            
        if scalars:
            self.scalars_csv = savecsv(file_path+"scalars.csv")
        else:
            self.scalars_csv = None
        
    def export(self):
        if self.positions_csv:
            self.positions_csv.write_row(self.positions)
            
        if self.directions_csv:
            self.directions_csv.write_row(self.directions)

        if self.centre_of_mass_csv:
            self.centre_of_mass_csv.write_row(self.centre_of_mass)
            
        if self.scalars_csv:
            self.scalars_csv.write_row([self.average_nearest_neighbour,
                                        self.average_distance_centre_of_mass])
        
    def close_files(self):
        if self.positions_csv:
            self.positions_csv.close_writer()
            
        if self.directions_csv:
            self.directions_csv.close_writer()
            
        if self.centre_of_mass_csv:
            self.centre_of_mass_csv.close_writer()
        
        if self.scalars_csv:
            self.scalars_csv.close_writer()
    
    @property
    def positions(self):
        return np.ndarray.flatten(np.array([b.position for b in self.boids]))
    
    @property
    def directions(self):
        return np.ndarray.flatten(np.array(
            [b.velocity/np.linalg.norm(b.velocity) for b in self.boids]))
    
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
