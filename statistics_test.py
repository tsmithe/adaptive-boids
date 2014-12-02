import statistics
import savecsv
import boids
from scipy import spatial
import numpy as np

population = []
for i in range(1,10):
        population.append(boids.Prey(100))

positions = np.array([b.position for b in population])
tree = spatial.cKDTree(positions)

file_path = "your_path"

stats = statistics.statistics(population, tree, file_path, position = True, direction = True)

for i in range(1,100):
    stats.export
    
# You now have the data you specified in the directory you specified.

import loadcsv

instance = loadcsv.loadcsv()
instance.load_file(file_path+"positions.csv",2)