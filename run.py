#!/usr/bin/env python3

"""
Run the main loop of the simulation
"""

import configparser, os, sys
import numpy as np

from boids import *
from statistics import StatisticsHelper


def export_stats(prey_statistics, predator_statistics, ecosystem):
    prey_statistics.update_data(ecosystem.prey, ecosystem.prey_tree)
    prey_statistics.export()
    predator_statistics.update_data(ecosystem.predators,
                                    ecosystem.predator_tree)
    predator_statistics.export()


def main(config):
    np.random.seed(eval(config['DEFAULT']['seed']))
    dt = eval(config['DEFAULT']['dt'])
    run_time = eval(config['DEFAULT']['run_time'])
    dump_stats_interval = eval(config['DEFAULT']['dump_stats_interval'])
    flush_files_interval = eval(config['DEFAULT'].get('flush_files_interval', 'np.inf'))

    ecosystem = Ecosystem(config)

    prey_statistics = StatisticsHelper('prey_', config)
    predator_statistics = StatisticsHelper('predator_', config)

    t = 0
    iteration = 0
    while t < run_time:
        sys.stdout.write("\rt = %.1f" % t)
        sys.stdout.flush()

        # Collision check
        #  -- predators collided with prey are marked as feeding
        #  -- prey collided with feeding area marked as feeding
        #  -- collisions induce velocity change

        # Feeding
        #  -- those boids marked as feeding receive a linear(?) increase in stamina
        ecosystem.update_stamina()

        # Kill and spawn
        #  -- check for boids marked as dead, and respawn them
        #  -- update lifespan values for living boids
        ecosystem.kill_prey()
        ecosystem.kill_predators()
        ecosystem.update_age()

        # Compute statistics and dump data to disk
        if not iteration % dump_stats_interval:
            export_stats(prey_statistics, predator_statistics, ecosystem)
        if not iteration % flush_files_interval:
            prey_statistics.flush()
            predator_statistics.flush()

        # Escape criterion: if avg life span not increasing for large no. of runs

        # Update time
        t += dt
        iteration += 1

        # Update positions (and corresponding regions) and stamina
        ecosystem.update_boid_velocities()
        ecosystem.update_velocity_data()
        ecosystem.update_boid_positions()
        ecosystem.update_position_data()

        # ...

    sys.stdout.write("\n")


if __name__ == '__main__':
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

    lock = os.path.join(config['DEFAULT']['data_dir'], 'lock')
    if os.path.exists(lock):
        print("You need to stop the running simulation, or remove the lock file at\n%s" %
              os.path.join(config['DEFAULT']['data_dir'], 'lock'))
        sys.exit(1)
    with open(lock, 'a'):
        os.utime(lock)

    try:
        main(config)
    except KeyboardInterrupt:
        pass

    os.remove(lock)
