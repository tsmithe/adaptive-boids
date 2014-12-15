#!/usr/bin/env python3

import configparser, csv, os, sys
import matplotlib.pyplot as plt
import numpy as np

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

def collect_data(csv_reader):
    frame_data = []
    i = 0
    for row in csv_reader:
        frame_data.append(np.array(list(map(lambda x: float(x), row))))
        
    return np.transpose(frame_data)

data = collect_data(csv_reader)

plt.plot(data[0])
plt.title("Average distance to nearest neighbour")
plt.figure()
plt.plot(data[1])
plt.title("Average distance to centre of mass")
plt.figure()
plt.title("Angular deviation")
plt.plot(data[2])
plt.show()
