#!/usr/bin/env python3

"""
Script to produce visualisation of boid movement from dumped position data
"""

# TODO:
# - mark boundary; feeding area
# - perhaps velocity arrows (optionally)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv, numpy as np

PLOT_MINIMUM = -400
PLOT_MAXIMUM = 400

def animate(i, prey_graph, predator_graph, prey_frames, predator_frames):
    prey_graph.set_offsets(prey_frames[i])
    predator_graph.set_offsets(predator_frames[i])
    return [prey_graph, predator_graph]

def collect_data(csv_reader):
    frame_data = []
    for row in csv_reader:
        positions = np.array(list(map(lambda x: float(x), row)))
        num_boids = positions.size / 2
        positions = positions.reshape(num_boids, 2)
        data = []
        for boid in positions:
            data.append((boid[0], boid[1]))
        frame_data.append(data)
    return frame_data

prey_reader = csv.reader(open('prey_positions.csv'))
predator_reader = csv.reader(open('predator_positions.csv'))

fig, ax = plt.subplots()
prey_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM, color='green')
predator_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM, color='red')
ax.set_xlim(PLOT_MINIMUM, PLOT_MAXIMUM)
ax.set_ylim(PLOT_MINIMUM, PLOT_MAXIMUM)

prey_frame_data = collect_data(prey_reader)
predator_frame_data = collect_data(predator_reader)

ani = animation.FuncAnimation(fig, animate, len(prey_frame_data),
                              fargs=(prey_graph, predator_graph,
                                     prey_frame_data, predator_frame_data),
                              interval=100,
                              repeat=True)
plt.show()

