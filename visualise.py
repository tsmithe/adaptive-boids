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

from run import DUMP_STATS_INTERVAL, DT, WORLD_RADIUS, PREY_RADIUS, PREDATOR_RADIUS

PLOT_MINIMUM = -2*WORLD_RADIUS
PLOT_MAXIMUM = 2*WORLD_RADIUS

FRAME_INTERVAL = 50 # NB: Matplotlib doesn't seem to go much faster...

def animate(i, fig, ax, text, prey_graph, predator_graph,
            prey_frames, predator_frames):
    text.set_text("t = %d" % (i*(DUMP_STATS_INTERVAL*DT)))
    prey_graph.set_offsets(prey_frames[i])
    predator_graph.set_offsets(predator_frames[i])
    return [text, prey_graph, predator_graph]

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
boundary = plt.Circle((0, 0), WORLD_RADIUS, facecolor='none',
                      linestyle='dashed')
ax.add_artist(boundary)
text = ax.text(PLOT_MINIMUM+5, PLOT_MAXIMUM-20, "", withdash=True, fontsize=12)
prey_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM,
                        np.pi*PREY_RADIUS**2, color='green')
predator_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM,
                            np.pi*PREDATOR_RADIUS**2, color='red')
ax.set_xlim(PLOT_MINIMUM, PLOT_MAXIMUM)
ax.set_ylim(PLOT_MINIMUM, PLOT_MAXIMUM)

prey_frame_data = collect_data(prey_reader)
predator_frame_data = collect_data(predator_reader)

ani = animation.FuncAnimation(fig, animate, len(prey_frame_data),
                              fargs=(fig, ax, text,
                                     prey_graph, predator_graph,
                                     prey_frame_data, predator_frame_data),
                              interval=FRAME_INTERVAL,
                              repeat=True)
plt.show()

