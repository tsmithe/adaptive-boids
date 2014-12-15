import csv, os
import numpy as np

class StatisticsHelper:
    
    def __init__(self, file_path, config, append_to_file = False,
                 position = False,
                 velocities = False,
                 centre_of_mass = False,
                 scalars = False,
                 ):
        '''
        The named parameters correspond to statistical numbers that can be
        exported. They will only be exported if the parameter is set to true.
        '''

        os.makedirs(config['DEFAULT']['data_dir'], exist_ok=True)
        
        if append_to_file: self.file_mode = 'a'
        else: self.file_mode = 'w'
        
        if position:
            fname = os.path.join(config['DEFAULT']['data_dir'], file_path+'positions.csv')
            self.positions_csv = csv.writer(open(fname, self.file_mode))
        else:
            self.positions_csv = None
            
        if velocities:
            fname = os.path.join(config['DEFAULT']['data_dir'], file_path+'velocities.csv')
            self.velocities_csv = csv.writer(open(fname, self.file_mode))
        else:
            self.velocities_csv = None

        if centre_of_mass:
            fname = os.path.join(config['DEFAULT']['data_dir'], file_path+'centre_of_mass.csv')
            self.centre_of_mass_csv = csv.writer(open(fname, self.file_mode))
        else:
            self.centre_of_mass_csv = None
            
        if scalars:
            fname = os.path.join(config['DEFAULT']['data_dir'], file_path+'scalars.csv')
            self.scalars_csv = csv.writer(open(fname, self.file_mode))
        else:
            self.scalars_csv = None

    def update_data(self, boids, tree):
        self.boids = boids
        self.tree = tree
        self.self_indices, self.neighbour_indices = self.find_nearest_neighbour()
    

    def export(self):
        if self.positions_csv:
            self.positions_csv.writerow(self.positions.tolist())
            
        if self.velocities_csv:
            self.velocities_csv.writerow(self.velocities.tolist())

        if self.centre_of_mass_csv:
            self.centre_of_mass_csv.writerow(self.centre_of_mass.tolist())
        
        if self.scalars_csv:
            self.scalars_csv.writerow([self.average_nearest_neighbour,
                                       self.average_distance_centre_of_mass,
                                       self.angular_deviation])

    @property
    def positions(self):
        return np.ndarray.flatten(np.array([b.position for b in self.boids]))
    
    @property
    def velocities(self):
        return np.ndarray.flatten(np.array([b.velocity for b in self.boids]))
    
    @property
    def average_nearest_neighbour(self):
        # Closest neighbour
        distances_temp, indices_temp = self.tree.query([b.position for b in self.boids], k=2)
        distances_temp = np.array(distances_temp)
        distances = np.zeros(np.size(distances_temp)/2)

        for i in np.arange(np.size(distances_temp)/2):
            if (distances_temp[i,0] == 0.0):
                distances[i] = distances_temp[i,1]
            elif (distances_temp[i,1] == 0.0):
                distances[i] = distances_temp[i,0]
            else:
                if (distances_temp[i,0] < distances_temp[i,1]):
                    distances[i] = distances_temp[i,0]
                else:
                    distances[i] = distances_temp[i,1]

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
    
    @property
    def angular_deviation(self):
        # Implements the measure described here: http://en.wikipedia.org/wiki/Mean_of_circular_quantities#Mean_of_angles
        # 0 means the velocity vectors point in randomly distributed directions
        # 1 means the velocity vectors are coordinated in the same direction
        
        #mean = np.array([0,0])
        boid_velocities = np.array([b.velocity for b in self.boids])
        angular_cosine = 0.0

        for i in np.arange(len(self.self_indices)):
            self_velocity = boid_velocities[self.self_indices[i],:]
            self_direction = self_velocity/np.linalg.norm(self_velocity)
            neighbour_velocity = boid_velocities[self.neighbour_indices[i],:]
            neighbour_direction = neighbour_velocity/np.linalg.norm(neighbour_velocity)
            angular_cosine += np.dot(self_direction, neighbour_direction)
        """
        for b in self.boids:
            self_velocity = boid_velocities[
            mean = np.add(mean,b.velocity/np.linalg.norm(b.velocity))
        return np.linalg.norm(mean/len(self.boids))
        """
        return angular_cosine/len(self.self_indices)

    def find_nearest_neighbour(self):
        distances, indices = self.tree.query([b.position for b in self.boids], k=2)
        self_indices = []
        neighbour_indices = []

        for i in np.arange(np.size(indices)/2):
            if (distances[i,0] == 0.0):
                self_indices.append(indices[i,0])
                neighbour_indices.append(indices[i,1])
            elif (distances[i,1] == 0.0):
                self_indices.append(indices[i,1])
                neighbour_indices.append(indices[i,1])
            else:
                if (distances[i,0] < distances[i,1]):
                    self_indices.append(indices[i,1])
                    neighbour_indices.append(indices[i,0])
                else:
                    self_indices.append(indices[i,0])
                    neighbour_indices.append(indices[i,1])
        
        return self_indices, neighbour_indices





