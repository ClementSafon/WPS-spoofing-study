""" Main file of the project. """
import time
import numpy as np
import graph
from radio_map import RadioMap
import knn_algorithm as knn

def simu1_display(error_positions, error_floors):
    """ Display the results of the simulation 1."""
    print(" -> location:")
    print("# Mean error: ", np.mean(error_positions))
    print("# Standard deviation: ", np.std(error_positions))
    print("# Max error: ", np.max(error_positions))
    print("# Min error: ", np.min(error_positions))
    print("# Median error: ", np.median(error_positions))
    print("# 25th percentile: ", np.percentile(error_positions, 25))
    print("# 75th percentile: ", np.percentile(error_positions, 75))
    print("# 90th percentile: ", np.percentile(error_positions, 90))
    print("# 95th percentile: ", np.percentile(error_positions, 95))
    print("# 99th percentile: ", np.percentile(error_positions, 99))
    print("# 99.99th percentile: ", np.percentile(error_positions, 99.99))
    print(" -> Floor:")
    print("# Mean error: ", np.mean(error_floors))
    print("# Standard deviation: ", np.std(error_floors))
    print("# Max error: ", np.max(error_floors))
    print("# Min error: ", np.min(error_floors))
    print("# Median error: ", np.median(error_floors))

def simu1_find_error_by_method(n_neighbors=3, method='normal'):
    """ Find the mean error by method (normal or log_penality or both)."""
    trning_r_m = RadioMap()
    trning_r_m.load('data/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')

    if method == 'normal':
        error_positions, error_floors = [ knn.run(
        n_neighbors, False, False, vld_r_m, False)[i] for i in (1, 3)]
        print("##################################################")
        print("Log penality: False")
        simu1_display(error_positions, error_floors)
    elif method == 'log_penality':
        error_positions, error_floors = [ knn.run(
        n_neighbors, False, False, vld_r_m, True)[i] for i in (1, 3)]
        print("##################################################")
        print("Log penality: True")
        simu1_display(error_positions, error_floors)
    elif method == 'both':
        error_positions, error_floors = [ knn.run(
        n_neighbors, False, False, vld_r_m, False)[i] for i in (1, 3)]
        print("##################################################")
        print("Log penality: False")
        simu1_display(error_positions, error_floors)
        error_positions, error_floors = [ knn.run(
        n_neighbors, False, False, vld_r_m, True)[i] for i in (1, 3)]
        print("##################################################")
        print("Log penality: True")
        simu1_display(error_positions, error_floors)


def simu2_spoofing(spoofed_row, spoofing_row):
    """ Simulate the spoofing."""
    trning_r_m = RadioMap()
    trning_r_m.load('data/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')
    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofing_row])

    confidence_value = 16.12

    trgt_r_m.spoof(spoofed_row, vld_r_m, spoofing_row)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)[0]

    title = f"Spoofing with valid fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m, title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(vld_r_m.fork([spoofing_row]).get_positions_by_row(spoofing_row
                                    ), args='yo', label='Position of the point that is spoofed')
    graph.plot_point(trgt_r_m.get_positions()[0], args='go')
    graph.plot_confidence_circle(average_positions[0], confidence_value)
    graph.show()


def sim3_randoom_spoofing(spoofed_row, seed):
    """ Simulate the random spoofing."""
    trning_r_m = RadioMap()
    trning_r_m.load('data/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')
    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofed_row])

    confidence_value = 16.12

    trgt_r_m.random_spoof(0, seed)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)

    title = f"Spoofing with completly random fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m,
                      title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(trgt_r_m.get_positions_by_row(0), args='go', label='Real position')
    graph.plot_confidence_circle(average_positions[0], confidence_value)
    graph.show()


def simu4_all_spoofing(spoofed_row, spoofing_row, seed):
    """ Simulate the spoofing."""
    trning_r_m = RadioMap()
    trning_r_m.load('data/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')
    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofing_row])

    confidence_value = 16.12

    trgt_r_m.spoof(spoofed_row, vld_r_m, spoofing_row)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)[0]

    title = f"Spoofing with valid fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m, title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(vld_r_m.fork([spoofing_row]).get_positions_by_row(spoofing_row
                                    ), args='yo', label='Position of the point that is spoofed')
    graph.plot_point(trgt_r_m.get_positions()[0], args='go')
    graph.plot_confidence_circle(average_positions[0], confidence_value)

    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofed_row])

    confidence_value = 16.12

    trgt_r_m.random_spoof(0, seed)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)

    title = f"Spoofing with completly random fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m,
                      title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(trgt_r_m.get_positions_by_row(0), args='go', label='Real position')
    graph.plot_confidence_circle(average_positions[0], confidence_value)
    graph.show()


if __name__ == '__main__':
    td = time.time()

    # Simulation 1
    # simu1_find_error_by_method(3, 'both')

    # Simulation 2
    # simu2_spoofing(0, 50)

    # Simulation 3
    # sim3_randoom_spoofing(0, 123)

    # Simulation 4
    simu4_all_spoofing(0, 50, 123)

    print("Executed in ", time.time() - td, " seconds")
