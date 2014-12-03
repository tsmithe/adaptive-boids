#!/usr/bin/env python3

"""
Script to produce visualisation of boid movement from dumped position data
"""

# TODO!

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(i, prey_graph, predator_graph, prey_frames, predator_frames):
    prey_graph.set_offsets(prey_frames[i])
    predator_graph.set_offsets(predator_frames[i])
    return [prey_graph, predator_graph]

#from loadcsv import loadcsv
import csv, numpy as np
prey_reader = csv.reader(open('prey_positions.csv'))
predator_reader = csv.reader(open('predator_positions.csv'))

fig, ax = plt.subplots()
prey_graph = ax.scatter(-2, -2, color='green')
predator_graph = ax.scatter(-2, -2, color='red')
ax.set_xlim(-1.5, 101.5)
ax.set_ylim(-1.5, 101.5)
prey_frame_data, predator_frame_data = [], []

for row in prey_reader:
    positions = np.array(list(map(lambda x: float(x), row)))
    num_boids = positions.size / 2
    positions = positions.reshape(num_boids, 2)
    data = []
    for boid in positions:
        data.append((boid[0], boid[1]))
    prey_frame_data.append(data)

for row in predator_reader:
    positions = np.array(list(map(lambda x: float(x), row)))
    num_boids = positions.size / 2
    positions = positions.reshape(num_boids, 2)
    data = []
    for boid in positions:
        data.append((boid[0], boid[1]))
    predator_frame_data.append(data)
    
ani = animation.FuncAnimation(fig, animate, len(prey_frame_data),
                              fargs=(prey_graph, predator_graph,
                                     prey_frame_data, predator_frame_data),
                              interval=100)#,
                              #blit=True, repeat=True)
plt.show()

