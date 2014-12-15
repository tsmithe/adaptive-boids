#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np

csv_reader = csv.reader(open('prey_scalars.csv'))

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
