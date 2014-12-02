from savecsv import savecsv
import numpy as np

class statistics:
    
    def __init__(self, boids, tree, file_path,
                 position = False,
                 direction = False,
                 average_nearest_neighbour = False,
                 center_of_mass = False,
                 average_distance_center_of_mass = False
                 ):
        '''
        The named parameters correspond to statistical numbers that can be
        exported. They will only be exported if the parameter is set to true.
        '''
        self.boids = boids
        self.tree = tree
        
        if position:
            self.positions = savecsv(file_path+"positions.csv")
        else:
            self.positions = False
            
        if direction:
            self.directions = savecsv(file_path+"directions.csv")
        else:
            self.directions = False
            
        if average_nearest_neighbour:
            self.average_nearest_neighbours = savecsv(file_path+"average_nearest_neighbours.csv")
        else:
            self.average_nearest_neighbours = False
        
        if center_of_mass:
            self.center_of_mass = savecsv(file_path+"center_of_mass.csv")
        else:
            self.center_of_mass = False
        
        if average_distance_center_of_mass:
            self.average_distance_center_of_mass = savecsv(file_path+"average_distance_to_center_of_mass.csv")
        else:
            self.average_distance_center_of_mass = False
        
    @property
    def export(self):
        if self.positions:
            self.positions.write_row(self.gather_positions)
            
        if self.directions:
            self.directions.write_row(self.compute_directions)
            
        if self.average_nearest_neighbours:
            self.average_nearest_neighbours.write_row(self.comppute_average_nearest_neighbour)
        
        if self.center_of_mass:
            self.center_of_mass.write_row(self.compute_center_of_mass)
        
        if self.average_distance_center_of_mass:
            self.average_distance_center_of_mass.write_row(self.compute_average_distance_to_center_of_mass)
        
    def close_files(self):
        if self.positions:
            self.positions.close_writer()
            
        if self.directions:
            self.directions.close_writer()
            
        if self.average_nearest_neighbours:
            self.average_nearest_neighbours.close_writer()
        
        if self.center_of_mass:
            self.center_of_mass.close_writer()
        
        if self.average_distance_center_of_mass:
            self.average_distance_center_of_mass.close_writer()
    
    @property
    def gather_positions(self):
        return np.ndarray.flatten(np.array([b.position for b in self.boids]))
    
    @property
    def compute_directions(self):
        return np.ndarray.flatten(np.array(
            [b.velocity/np.linalg.norm(b.velocity) for b in self.boids]))
    
    @property
    def comppute_average_nearest_neighbour(self):
        # Closest neighbour
        distances, indices = self.tree.query([b.position for b in self.boids])
        return np.mean(distances)
    
    @property
    def compute_center_of_mass(self):
        # Computes the center of mass
        position_sum = np.array([0,0])
        for b in self.boids:
            position_sum += b.position
            
        return position_sum/len(self.boids)
    
    @property
    def compute_average_distance_center_of_mass(self):
        # Computes the average distance to the center of mass
        center_of_mass = self.center_of_mass
        distance_sum = 0;
        for b in self.boids:
            distance_sum += np.linalg.norm(b.position-center_of_mass)
            
        return distance_sum/len(boids)