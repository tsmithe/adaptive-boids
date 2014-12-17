#!/usr/bin/env python3

"""
Script to produce visualisation of boid movement from dumped position data
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import configparser, csv, numpy as np, os, sys

from fast_boids import quick_norm

from boids import *

if len(sys.argv) < 2:
    path = '.'
else:
    path = sys.argv[1]

config = configparser.ConfigParser()
if os.path.isdir(path):
    config.read(os.path.join(path, 'config.ini'))
    config['DEFAULT']['data_dir'] = os.path.abspath(path)
else:
    config.read(path)
    config['DEFAULT']['data_dir'] = os.path.abspath(os.path.dirname(path))

WORLD_RADIUS = eval(config['DEFAULT']['world_radius'])
DUMP_STATS_INTERVAL = eval(config['DEFAULT']['dump_stats_interval'])
DT = eval(config['DEFAULT']['dt'])

PREY_RADIUS = eval(config['Prey']['boid_radius'])
PREDATOR_RADIUS = eval(config['Predator']['boid_radius'])

FRAME_INTERVAL = eval(config['Visualisation']['frame_interval'])
EVERY_NTH_FRAME = eval(config['Visualisation']['every_nth_frame'])
START_AT_T = eval(config['Visualisation']['start_at_t'])
STOP_AT_T = eval(config['Visualisation']['stop_at_t'])

DATA_DIR = config['DEFAULT']['data_dir']

PLOT_MINIMUM = -1.01*WORLD_RADIUS
PLOT_MAXIMUM = 1.01*WORLD_RADIUS

feeding_area_helper = FeedingAreaConfigurations()
FEEDING_AREA_LOCATIONS, FEEDING_AREA_RADIUS = feeding_area_helper.get_info(eval(config['DEFAULT']['feeding_areas']))

def animate(i, fig, ax, text,
            prey_graph, prey_quivers,
            predator_graph, predator_quivers,
            prey_pos_data, prey_vel_data,
            predator_pos_data, predator_vel_data):
    text.set_text("t = %d" % (i*(DUMP_STATS_INTERVAL*DT*EVERY_NTH_FRAME)
                              + START_AT_T))
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
    i = 0
    for row in csv_reader:
        if i % EVERY_NTH_FRAME:
            i += 1
            continue
        if i*DUMP_STATS_INTERVAL*DT < START_AT_T:
            i += 1
            continue
        if i*DUMP_STATS_INTERVAL*DT > STOP_AT_T:
            break
        positions = np.array(list(map(lambda x: float(x), row)))
        num_boids = positions.size / 2
        positions = positions.reshape(num_boids, 2)
        data = []
        for boid in positions:
            data.append((boid[0], boid[1]))
        frame_data.append(data)
        i += 1
    return frame_data

prey_pos_reader = csv.reader(open(os.path.join(DATA_DIR, 'prey_positions.csv')))
prey_vel_reader = csv.reader(open(os.path.join(DATA_DIR, 'prey_velocities.csv')))
predator_pos_reader = csv.reader(open(os.path.join(DATA_DIR, 'predator_positions.csv')))
predator_vel_reader = csv.reader(open(os.path.join(DATA_DIR, 'predator_velocities.csv')))

prey_pos_data = collect_data(prey_pos_reader)
prey_vel_data = collect_data(prey_vel_reader)
predator_pos_data = collect_data(predator_pos_reader)
predator_vel_data = collect_data(predator_vel_reader)

fig, ax = plt.subplots()
boundary = plt.Circle((0, 0), WORLD_RADIUS, facecolor='none',
                      linestyle='dashed')
ax.add_artist(boundary)

for location in FEEDING_AREA_LOCATIONS:
    feeding_plot = plt.Circle(tuple(location), FEEDING_AREA_RADIUS, facecolor='green', linestyle='dashed', alpha=0.3)
    ax.add_artist(feeding_plot)

text = ax.text(PLOT_MINIMUM+5, PLOT_MAXIMUM-20, "", withdash=True, fontsize=12)

prey_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM,
                        1.2*np.pi*PREY_RADIUS**2, facecolor='green', alpha=0.8,
                        edgecolor='black', linewidth=1)
prey_quivers = ax.quiver([], [], width=0.5, units='dots', scale=0.08)
predator_graph = ax.scatter(100*PLOT_MINIMUM, 100*PLOT_MAXIMUM,
                            1.2*np.pi*PREDATOR_RADIUS**2, facecolor='red', alpha=0.8,
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

