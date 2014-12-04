# run `python3 setup.py build_ext --inplace` first !

from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport acos, pow, sqrt
from libc.stdlib cimport abort, malloc, free
from cython.parallel import parallel, prange

def find_visible_neighbours(np.ndarray[np.float64_t, ndim=2] positions,
                            np.ndarray[np.float64_t, ndim=1] my_position,
                            np.ndarray[np.float64_t, ndim=1] my_velocity,
                            double perception_angle,
                            list neighbour_indices):

    cdef set visible_neighbours = set()
    cdef int num_neighbours = len(neighbour_indices)
    cdef int idx = 0, j = 0

    cdef double my_velocity_x = my_velocity[0]
    cdef double my_velocity_y = my_velocity[1]

    cdef double my_velocity_norm = sqrt(pow(my_velocity_x, 2) +
                                        pow(my_velocity_y, 2))
    cdef double relative_norm = 0

    cdef double my_position_x = my_position[0]
    cdef double my_position_y = my_position[1]

    cdef double angle = 0

    cdef double* relative_xs = <double*> malloc(sizeof(double) * num_neighbours)
    cdef double* relative_ys = <double*> malloc(sizeof(double) * num_neighbours)
    cdef int* neighbour_array = <int*> malloc(sizeof(int) * num_neighbours)
    cdef int* visible_array = <int*> malloc(sizeof(int) * num_neighbours)

    # Initialize data that will be needed outside of Python..
    for j in range(num_neighbours):
        idx = neighbour_indices[j]
        relative_xs[j] = positions[idx, 0] - my_position_x
        relative_ys[j] = positions[idx, 1] - my_position_y
        visible_array[j] = -1
        neighbour_array[j] = idx

    # Run (parallel) code outside of Python
    with nogil:#, parallel(num_threads=4):
        for j in range(num_neighbours): #, schedule='static', chunksize=12, num_threads=6):
            idx = neighbour_array[j]
            #relative_xs[j] -= my_position_x
            #relative_ys[j] -= my_position_y
            if relative_xs[j] + relative_ys[j] == 0:
                continue

            relative_norm = sqrt(pow(relative_xs[j], 2) + pow(relative_ys[j], 2))
            angle = acos(sqrt(relative_xs[j] * relative_ys[j] + 
                              my_velocity_x * my_velocity_y)/
                         (relative_norm * my_velocity_norm))
            if angle < 0: angle = -angle

            if angle < perception_angle:
                visible_array[j] = idx

    # Copy visible neighbour indices back to a Python set
    for j in range(num_neighbours):
        if visible_array[j] < 0: continue
        visible_neighbours.add(visible_array[j])

    # Free non-Python data structures
    free(relative_xs)
    free(relative_ys)
    free(neighbour_array)
    free(visible_array)

    return list(visible_neighbours)
