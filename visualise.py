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
from fast_boids import quick_norm

PLOT_MINIMUM = -1.01*WORLD_RADIUS
PLOT_MAXIMUM = 1.01*WORLD_RADIUS

FRAME_INTERVAL = 50 # NB: Matplotlib doesn't seem to go much faster...

def animate(i, fig, ax, text,
            prey_graph, prey_quivers,
            predator_graph, predator_quivers,
            prey_pos_data, prey_vel_data,
            predator_pos_data, predator_vel_data):
    text.set_text("t = %d" % (i*(DUMP_STATS_INTERVAL*DT)))
    u = [u[0]/quick_norm(np.array(u)) for u in prey_vel_data[i]]
    v = [u[1]/quick_norm(np.array(u)) for u in prey_vel_data[i]]
    prey_quivers.set_offsets(prey_pos_data[i])
    prey_quivers.set_UVC(u, v)
    prey_graph.set_offsets(prey_pos_data[i])
    u = [u[0]/quick_norm(np.array(u)) for u in predator_vel_data[i]]
    v = [u[1]/quick_norm(np.array(u)) for u in predator_vel_data[i]]
    predator_quivers.set_offsets(predator_pos_data[i])
    predator_quivers.set_UVC(u, v)
    predator_graph.set_offsets(predator_pos_data[i])
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

prey_pos_reader = csv.reader(open('prey_positions.csv'))
prey_vel_reader = csv.reader(open('prey_velocities.csv'))
predator_pos_reader = csv.reader(open('predator_positions.csv'))
predator_vel_reader = csv.reader(open('predator_velocities.csv'))

prey_pos_data = collect_data(prey_pos_reader)
prey_vel_data = collect_data(prey_vel_reader)
predator_pos_data = collect_data(predator_pos_reader)
predator_vel_data = collect_data(predator_vel_reader)

fig, ax = plt.subplots()
boundary = plt.Circle((0, 0), WORLD_RADIUS, facecolor='none',
                      linestyle='dashed')
ax.add_artist(boundary)
text = ax.text(PLOT_MINIMUM+5, PLOT_MAXIMUM-20, "", withdash=True, fontsize=12)
prey_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM,
                        np.pi*PREY_RADIUS**2, facecolor='green', alpha=0.8,
                        edgecolor='black', linewidth=1)
prey_quivers = ax.quiver([], [], width=0.5, units='dots', scale=0.08)
predator_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM,
                            np.pi*PREDATOR_RADIUS**2, facecolor='red', alpha=0.8,
                            edgecolor='black', linewidth=1)
predator_quivers = ax.quiver([], [], width=0.5, units='dots', scale=0.08)
ax.set_xlim(PLOT_MINIMUM, PLOT_MAXIMUM)
ax.set_ylim(PLOT_MINIMUM, PLOT_MAXIMUM)

ani = animation.FuncAnimation(fig, animate, len(prey_pos_data),
                              fargs=(fig, ax, text,
                                     prey_graph, prey_quivers,
                                     predator_graph, predator_quivers,
                                     prey_pos_data, prey_vel_data,
                                     predator_pos_data, predator_vel_data),
                              interval=FRAME_INTERVAL,
                              repeat=True)
plt.show()

