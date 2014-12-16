#!/usr/bin/env python3

import configparser, csv, os, sys
import matplotlib.pyplot as plt
import numpy as np

MOVING_AVG = False

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
csv_reader = csv.reader(open(os.path.join(data_dir, 'prey_scalars.csv')))
try:
    prey_fitness_reader = csv.reader(open(os.path.join(data_dir, 'prey_fitness.csv')))
    predator_fitness_reader = csv.reader(open(os.path.join(data_dir, 'predator_fitness.csv')))
except FileNotFoundError:
    prey_fitness_reader = None
    predator_fitness_reader = None

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

def calculate_moving_average(data_vector, n):
    # Half point to the left, half to the right
    n_half = np.floor(0.5*(n-1))
    moving_average = np.zeros(len(data_vector))
    for i in np.arange(len(data_vector)):
        if (i-n_half < 0):
            moving_average[i] = np.mean(data_vector[0:i+n_half])
        elif (i+n_half > len(data_vector)):
            moving_average[i] = np.mean(data_vector[i-n_half:])
        else:
            moving_average[i] = np.mean(data_vector[i-n_half:i+n_half])
    return moving_average
    
def calculate_moving_std(data_vector, n):
    n_half = np.floor(0.5*(n-1))
    moving_std = np.zeros(len(data_vector))
    for i in np.arange(len(data_vector)):
        if (i-n_half < 0):
            moving_std[i] = np.std(data_vector[0:i+n_half])
        elif (i+n_half > len(data_vector)):
            moving_std[i] = np.std(data_vector[i-n_half:])
        else:
            moving_std[i] = np.std(data_vector[i-n_half:i+n_half])
    return moving_std    

data = collect_data(csv_reader)

# Time vector
time_vector = make_time_vector(data[0], dt, dump_stats_interval)

plt.plot(time_vector, data[0])
plt.title("Average distance to nearest neighbour")
plt.xlabel("Time")
plt.ylabel("Average distance")

plt.figure()
plt.plot(time_vector, data[1])
plt.title("Average distance to centre of mass")
plt.xlabel("Time")

plt.figure()
plt.plot(time_vector, data[2])
plt.title("Cosine of angular deviation")
plt.xlabel("Time")
plt.ylabel("Angular deviation")

if MOVING_AVG:
    # Moving average and std of nearest-neighbour distance, keep n odd
    nn_dist_moving_avg = calculate_moving_average(data[0],501)
    nn_dist_moving_std = calculate_moving_std(data[0],501)

    # Moving average and std of nearest-neighbour angular deviation, keep n odd
    ang_dev_moving_avg = calculate_moving_average(data[2],501)
    ang_dev_moving_std = calculate_moving_std(data[2],501)

    plt.figure()
    plt.plot(time_vector, nn_dist_moving_avg)
    plt.title("Average distance to nearest neighbour, moving average of nearest 500 points")
    plt.xlabel("Time")
    plt.ylabel("Average distance")

    plt.figure()
    plt.plot(time_vector, ang_dev_moving_avg)
    plt.title("Cosine of angular deviation, moving averaging of the nearest 500 points")
    plt.xlabel("Time")
    plt.ylabel("Angular deviation")

    plt.figure()
    plt.plot(time_vector, nn_dist_moving_avg/np.max(nn_dist_moving_avg), time_vector, ang_dev_moving_avg)

if prey_fitness_reader and predator_fitness_reader:
    prey_fitness_data = np.average(collect_data(prey_fitness_reader), 0)
    predator_fitness_data = np.average(collect_data(predator_fitness_reader), 0)

    prey_fitness_moving_avg = calculate_moving_average(prey_fitness_data, 501)
    predator_fitness_moving_avg = calculate_moving_average(predator_fitness_data, 501)

    plt.figure()
    time_vector = make_time_vector(prey_fitness_data, dt, dump_stats_interval)
    plt.plot(time_vector, prey_fitness_moving_avg)
    plt.title("Prey fitness; moving average with window size 500")
    plt.xlabel("Time")
    plt.ylabel("Fitness")

    plt.figure()
    time_vector = make_time_vector(predator_fitness_data, dt, dump_stats_interval)
    plt.plot(time_vector, predator_fitness_moving_avg)
    plt.title("Predator fitness; moving average with window size 500")
    plt.xlabel("Time")
    plt.ylabel("Fitness")

plt.show()
