#!/usr/bin/env python3

import configparser, csv, os, sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

DPI = 150

MOVING_AVG = True

prey_weight_labels = {
    0: 'Fellow prey positions',
    1: 'Fellow prey velocities',
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

SAVE = False
if len(sys.argv) > 2:
    if sys.argv[2] == '-s':
        SAVE = True

FIGURES = [] # List of figures and filenames to save if necessary
ANALYSIS = ""
    
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

try:
    prey_avg_weights_reader = csv.reader(open(os.path.join(data_dir, 'prey_avg_weights.csv')))
    predator_avg_weights_reader = csv.reader(open(os.path.join(data_dir, 'predator_avg_weights.csv')))
except FileNotFoundError:
    prey_avg_weights_reader = None
    predator_avg_weights_reader = None

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

def plot_weights(csv_reader, title, weight_labels):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Weight')
    data = collect_data(csv_reader)
    for i in range(data.shape[0]):
        time_vector = make_time_vector(data[i,:], dt, dump_stats_interval)
        ax.plot(time_vector, data[i,:], label=weight_labels[i])
    ax.set_xlim(time_vector[0], time_vector[-1])
    plt.legend(fancybox=True, fontsize='small', loc='upper left', framealpha=0.8)
    return fig

print('Reading data...')
data = collect_data(csv_reader)

# Time vector
time_vector = make_time_vector(data[0], dt, dump_stats_interval)

print('Plotting...')

fig1, ax1 = plt.subplots()
ax1.plot(time_vector, data[0])
ax1.set_title("Average distance to nearest neighbour")
ax1.set_xlabel("Time")
ax1.set_ylabel("Average distance")
ax1.set_xlim(time_vector[0], time_vector[-1])
FIGURES.append((fig1, 'avg-nn-dist.png'))

fig2, ax2 = plt.subplots()
ax2.plot(time_vector, data[1])
ax2.set_title("Average distance to centre of mass")
ax2.set_xlabel("Time")
ax2.set_xlim(time_vector[0], time_vector[-1])
FIGURES.append((fig2, 'avg-centroid-dist.png'))

fig3, ax3 = plt.subplots()
ax3.plot(time_vector, data[2])
ax3.set_title("Cosine of angular deviation")
ax3.set_xlabel("Time")
ax3.set_ylabel("Angular deviation")
ax3.set_xlim(time_vector[0], time_vector[-1])
FIGURES.append((fig3, 'cosine-deviation.png'))

if MOVING_AVG:
    print('Computing moving averages for nn. distance and angular deviation...')

    try:
        moving_average_window_size = eval(config['Visualisation']['running_average_window_size'])
    except KeyError:
        moving_average_window_size = 501
    
    # Moving average and std of nearest-neighbour distance, keep n odd
    nn_dist_moving_avg = calculate_moving_average(data[0],moving_average_window_size)
    nn_dist_moving_std = calculate_moving_std(data[0], moving_average_window_size)

    # Moving average and std of nearest-neighbour angular deviation, keep n odd
    ang_dev_moving_avg = calculate_moving_average(data[2],moving_average_window_size)
    ang_dev_moving_std = calculate_moving_std(data[2],moving_average_window_size)

    print('Plotting...')

    fig4, ax4 = plt.subplots()
    ax4.plot(time_vector, nn_dist_moving_avg)
    ax4.set_title("Average distance to nearest neighbour; moving average with window size %d" % (moving_average_window_size-1))
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Average distance")
    ax4.set_xlim(time_vector[0], time_vector[-1])
    FIGURES.append((fig4, 'moving-avg_avg-nn-dist.png'))

    fig5, ax5 = plt.subplots()
    ax5.plot(time_vector, ang_dev_moving_avg)
    ax5.set_title("Cosine of angular deviation; moving average with window size %d" % (moving_average_window_size-1))
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Angular deviation")
    ax5.set_xlim(time_vector[0], time_vector[-1])
    FIGURES.append((fig5, 'moving-avg_cosine-deviation.png'))

    fig6, ax6 = plt.subplots()
    normed_dist_avg = nn_dist_moving_avg/nn_dist_moving_avg[0]
    combined_handle = ax6.plot(time_vector, normed_dist_avg, time_vector, ang_dev_moving_avg)
    plt.legend(combined_handle, ['Normalised nn distance','Angular deviation'])
    ax6.set_xlim(time_vector[0], time_vector[-1])
    FIGURES.append((fig6, 'moving-avg_nn-dist-deviation-comparison.png'))

    rho, p = scipy.stats.pearsonr(normed_dist_avg, ang_dev_moving_avg)
    ANALYSIS += "Distance-deviation correlation (Spearman; moving avg): %g with p-value %f\n" % (rho, p)

try:
    fig7, ax7 = plt.subplots()
    ax7.set_title("Average age and life-span of prey")
    ax7.plot(time_vector, data[3], label='Average age')
    ax7.plot(time_vector, data[4], label='Average life-span')
    ax7.set_xlabel('Time')
    plt.legend(fancybox=True, loc='upper left', fontsize='small')
    ax7.set_xlim(time_vector[0], time_vector[-1])
    FIGURES.append((fig7, 'avg-prey-age-lifespan.png'))
except IndexError: pass
    
rho, p = scipy.stats.pearsonr(data[0]/np.max(data[0]), data[2])
ANALYSIS += "Distance-deviation correlation (Spearman; full data): %g with p-value %f\n" % (rho, p)


if prey_fitness_reader and predator_fitness_reader:
    print('Reading fitness data...')
    
    prey_fitness_data = np.average(collect_data(prey_fitness_reader), 0)
    predator_fitness_data = np.average(collect_data(predator_fitness_reader), 0)

    print('Computing moving averages for fitness data...')

    prey_fitness_moving_avg = calculate_moving_average(prey_fitness_data, 501)
    predator_fitness_moving_avg = calculate_moving_average(predator_fitness_data, 501)

    print('Plotting...')
    
    fig8, ax8 = plt.subplots()
    time_vector = make_time_vector(prey_fitness_data, dt, dump_stats_interval)
    ax8.plot(time_vector, prey_fitness_moving_avg)
    ax8.set_title("Prey fitness; moving average with window size %d" % (moving_average_window_size-1))
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Fitness")
    ax8.set_xlim(time_vector[0], time_vector[-1])
    FIGURES.append((fig8, 'moving_avg-avg-prey-fitness.png'))

    fig9, ax9 = plt.subplots()
    time_vector = make_time_vector(predator_fitness_data, dt, dump_stats_interval)
    ax9.plot(time_vector, predator_fitness_moving_avg)
    ax9.set_title("Predator fitness; moving average with window size %d" % (moving_average_window_size-1))
    ax9.set_xlabel("Time")
    ax9.set_ylabel("Fitness")
    ax9.set_xlim(time_vector[0], time_vector[-1])
    FIGURES.append((fig9, 'moving_avg-avg-predator-fitness.png'))

if prey_avg_weights_reader and predator_avg_weights_reader:
    print('Plotting average weights...')
    fig10 = plot_weights(prey_avg_weights_reader, 'Prey weights', prey_weight_labels)
    FIGURES.append((fig10, 'prey-avg-weights.png'))
    fig11 = plot_weights(predator_avg_weights_reader, 'Predator weights', predator_weight_labels)
    FIGURES.append((fig11, 'predator-avg-weights.png'))
    
print('Saving analysis...')
with open(os.path.join(data_dir, 'analysis.txt'), 'wt') as analysis_file:
    analysis_file.write(ANALYSIS)

print('Displaying analysis:')
sys.stdout.write(ANALYSIS)

if not SAVE:
    print('Displaying figures...')
    plt.show()
else:
    print('Saving figures...')
    for fig in FIGURES:
        fig[0].savefig(os.path.join(data_dir, fig[1]), dpi=DPI, bbox_inches='tight')
        print('Done: %s' % fig[1])
