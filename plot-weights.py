#!/usr/bin/env python3

import configparser, csv, os, sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

prey_weight_labels = {
    0: 'Prey positions',
    1: 'Prey velocities',
    2: 'Boid too close',
    3: 'Predator position(s)',
    4: 'Boundary proximity',
    5: 'Feeding area position'
}

predator_weight_labels = {
    0: 'Prey positions',
    1: 'Prey velocities',
    2: 'Prey tracking',
    3: 'Predator position(s)',
    4: 'Predator velocities',
    5: 'Boid too close',
    6: 'Boundary proximity'
}

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

data_dir = config['DEFAULT']['data_dir']
prey_reader = csv.reader(open(os.path.join(data_dir, 'prey_avg_weights.csv')))
predator_reader = csv.reader(open(os.path.join(data_dir, 'predator_avg_weights.csv')))

dt = eval(config['DEFAULT']['dt'])
dump_stats_interval = eval(config['DEFAULT']['dump_stats_interval'])

def collect_data(csv_reader):
    frame_data = []
    i = 0
    for row in csv_reader:
        frame_data.append(np.array(list(map(lambda x: float(x), row))))
        
    return np.transpose(frame_data)
    
def make_time_vector(data_vector, dt, dump_stats_interval):
    time_between_dumps = dt*dump_stats_interval
    time_vector = time_between_dumps*np.arange(len(data_vector))
    return time_vector

def plot_weights(csv_reader, title, weight_labels):
    plt.figure()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Weight')
    data = collect_data(csv_reader)
    for i in range(data.shape[0]):
        plt.plot(make_time_vector(data[i,:], dt, dump_stats_interval),
                 data[i,:],
                 label=weight_labels[i])
    plt.legend(fancybox=True, fontsize='small', loc='upper left', framealpha=0.8)

plot_weights(prey_reader, 'Prey weights', prey_weight_labels)
plot_weights(predator_reader, 'Predator weights', predator_weight_labels)

plt.show()


