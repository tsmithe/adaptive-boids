#!/usr/bin/env python3

"""
Script to produce visualisation of boid movement from dumped position data
"""

# TODO!

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(i, graph, ax, frames):
    graph.set_offsets(frames[i])
    return graph

#from loadcsv import loadcsv
import csv, numpy as np
reader = csv.reader(open('prey_positions.csv'))

fig, ax = plt.subplots()
graph = ax.scatter(-2, -2)
ax.set_xlim(-1.5, 101.5)
ax.set_ylim(-1.5, 101.5)
frame_data = []

for row in reader:
    positions = np.array(list(map(lambda x: float(x), row)))
    num_boids = positions.size / 2
    positions = positions.reshape(num_boids, 2)
    data = []
    for boid in positions:
        data.append((boid[0], boid[1]))
    frame_data.append(data)

ani = animation.FuncAnimation(fig, animate, len(frame_data),
                              fargs=(graph, ax, frame_data),
                              interval=100,
                              blit=True, repeat=True)
plt.show()

