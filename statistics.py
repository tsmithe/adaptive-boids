import csv
import numpy as np

class statistics:
    save_positions = True
    save_velocities = True
    save_average_nearest_neighbour = True
    save_center_of_mass = True
    save_average_distance_to_center_of_mass = True
    
    def __init__(self, file_path, file_id, boids, tree):
        '''
        file_path should include a trailing slash
        existing files with the same file id will be overwritten
        '''
        self.file_base_name = file_path+file_id
        self.boids = boids
        self.tree = tree
        
        self.average_nearest_neighbour_file = None
        self.center_of_mass_file = None
        self.average_distance_to_center_of_mass_file = None
        
    @property
    def export(self):
        # Writes data to files in the CSV format
        
        # Positions
        if self.save_positions:
            if self.positions_file is None:
                self.positions_file = open(
                    self.file_base_name+'_positions','w')
                self.positions_writer = csv.writer(
                    positions_file)
        self.positions_writer.writerow(
            self.positions)
        
        # Velocities
        if self.save_velocities:
            if self.velocities_file is None:
                self.velocities_file = open(
                    self.file_base_name+'_positions','w')
                self.velocities_writer = csv.writer(
                    velocities_file)
        self.velocities_writer.writerow(
            self.velocities)
        
        # Average nearest neighbour
        if self.save_average_nearest_neighbour:
            if self.average_nearest_neighbour_file is None:
                self.average_nearest_neighbour_file = open(
                    self.file_base_name+'_average_nearest_neighbor.csv','w')
                self.average_nearest_neighbour_writer = csv.writer(
                    average_nearest_neighbour_file)
        self.average_nearest_neighbour_writer.writerow(
            self.average_nearest_neighbour)
    
        # Center of mass
        if self.save_center_of_mass:
            if self.center_of_mass_file is None:
                self.center_of_mass_file = open(
                    self.file_base_name+'_center_of_mass.csv','w')
                self.center_of_mass_writer = csv.writer(
                    center_of_mass_file)
        self.center_of_mass_writer.writerow(
            self.center_of_mass)
        
        # Average distance to center of mass
        if self.save_average_distance_to_center_of_mass:
            if self.average_distance_to_center_of_mass_file is None:
                self.average_distance_to_center_of_mass_file = open(
                    self.file_base_name+'average_distance_center_of_mass.csv','w')
                self.average_distance_to_center_of_mass_writer = csv.writer(
                    average_distance_to_center_of_mass_file)
        self.average_distance_to_center_of_mass_writer.writerow(
            self.average_distance_to_center_of_mass)
        
    def close_files(self):
        if not self.average_nearest_neighbour_file is None:
            self.average_nearest_neighbour_file.close()
            self.average_nearest_neighbour_file = None
        if not self.center_of_mass_file is None:
            self.center_of_mass_file.close()
            self.center_of_mass_file = None
        if not self.average_distance_to_center_of_mass_file is None:
            self.average_distance_to_center_of_mass_file.close()
            self.average_distance_to_center_of_mass_file = None
    
    @property
    def positions(self):
        return np.ndarray.flatten([b.position for b in self.boids])
    
    @property
    def directions(self):
        return np.ndarray.flatten(
            [b.velocity/np.linalg.norm(b.velocity) for b in self.boids])
    
    @property
    def average_nearest_neighbour(self):
        # Closest neighbour
        distances, indices = self.tree.query([b.position for b in self.boids])
        return np.mean(distances)
    
    @property
    def center_of_mass(self):
        # Computes the center of mass
        position_sum = np.array([0,0])
        for b in self.boids:
            position_sum += b.position
            
        return position_sum/len(self.boids)
    
    @property
    def average_distance_to_center_of_mass(self):
        # Computes the average distance to the center of mass
        center_of_mass = self.center_of_mass
        distance_sum = 0;
        for b in self.boids:
            distance_sum += np.linalg.norm(b.position-center_of_mass)
            
        return distance_sum/len(boids)