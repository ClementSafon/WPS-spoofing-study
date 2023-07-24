import knn_algorithm as knn
import matplotlib.pyplot as plt
import numpy as np
import time
from RadioMap import RadioMap
import signal
import sys
from Map import Map


def simu1_signal_handler(sig, frame):
    global best_n_neighbors
    best_n_neighbors = np.array(best_n_neighbors)
    for i in range(1, 10):
        print("n_neighbors = ", i, " : ", np.sum(best_n_neighbors == i))
    print("Best n_neighbors: ", np.argsort(np.bincount(best_n_neighbors))
          [len(np.bincount(best_n_neighbors)) - 1] + 1)
    sys.exit(0)


def simu1_find_best_k_neighbors():
    global best_n_neighbors
    signal.signal(signal.SIGINT, simu1_signal_handler)
    trainingRadioMap = RadioMap()
    trainingRadioMap.load('data/TrainingData.csv')
    validationRadioMap = RadioMap()
    validationRadioMap.load('data/ValidationData.csv')
    # seed = 1
    # n_neighbors = 1
    gui = False
    verbose = False
    best_n_neighbors = []
    for seed in range(1, 200):
        difference = []
        print("Seed: ", seed)
        for n_neighbors in range(1, 10):
            print("# n_neighbors: ", n_neighbors)
            difference.append(knn.run(seed, n_neighbors, gui,
                              verbose, trainingRadioMap)[1][0])
        difference = np.array(difference)
        best_n_neighbors.append(np.argsort(difference)[0]+1)
    best_n_neighbors = np.array(best_n_neighbors)
    for i in range(1, 10):
        print("n_neighbors = ", i, " : ", np.sum(best_n_neighbors == i))
    print("Best n_neighbors: ", np.argsort(np.bincount(best_n_neighbors))
          [len(np.bincount(best_n_neighbors)) - 1] + 1)


def simu3_find_error_mean():
    trainingRadioMap = RadioMap()
    trainingRadioMap.load('data/TrainingData.csv')
    validationRadioMap = RadioMap()
    validationRadioMap.load('data/ValidationData.csv')
    n_neighbors = 4
    gui = False
    verbose = False

    print("##################################################")
    print("Log penality: True")
    average_positions, error_positions, average_floors, error_floors = knn.run(
        0, n_neighbors, gui, verbose, trainingRadioMap, validationRadioMap, True)
    print("In Position:")
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

    print("In Floor:")
    print("# Mean error: ", np.mean(error_floors))
    print("# Standard deviation: ", np.std(error_floors))
    print("# Max error: ", np.max(error_floors))
    print("# Min error: ", np.min(error_floors))
    print("# Median error: ", np.median(error_floors))
    print("##################################################")

    print("##################################################")
    print("Log penality: False")
    average_positions, error_positions, average_floors, error_floors = knn.run(
        0, n_neighbors, gui, verbose, trainingRadioMap, validationRadioMap, False)
    print("In Position:")
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

    print("In Floor:")
    print("# Mean error: ", np.mean(error_floors))
    print("# Standard deviation: ", np.std(error_floors))
    print("# Max error: ", np.max(error_floors))
    print("# Min error: ", np.min(error_floors))
    print("# Median error: ", np.median(error_floors))
    print("##################################################")


def simu2_spoofing():
    trainingRadioMap = RadioMap()
    trainingRadioMap.load('data/TrainingData.csv')
    validationRadioMap = RadioMap()
    validationRadioMap.load('data/ValidationData.csv')
    targetRadioMap = validationRadioMap.fork([0, 1])

    targetRadioMap.spoof(0, trainingRadioMap, 200)
    targetRadioMap.spoof(1, validationRadioMap, 13)
    average_positions, errors = knn.run(
        0, 3, False, True, trainingRadioMap, targetRadioMap, log_penality=False)[:2]
    map = Map()

    map.plot_radioMap(trainingRadioMap,
                      title="Training data and test point", new_figure=True)
    map.plot_point(average_positions[0], args='ro')
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap,
                   trainingRadioMap.fork([0]), log_penality=False)[0][0], args='yo')
    map.plot_point(targetRadioMap.get_positions()[0], args='go')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[0], confidence_value)

    map.plot_radioMap(trainingRadioMap,
                      title="Training data and test point", new_figure=True)
    map.plot_point(average_positions[1], args='ro')
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap,
                   validationRadioMap.fork((1)), log_penality=False)[0][0], args='yo')
    map.plot_point(targetRadioMap.get_positions()[1], args='go')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[1], confidence_value)

    map.show()


def simu4_randoom_spoofing():
    trainingRadioMap = RadioMap()
    trainingRadioMap.load('data/TrainingData.csv')
    validationRadioMap = RadioMap()
    validationRadioMap.load('data/ValidationData.csv')
    start_row = 157
    targetRadioMap = validationRadioMap.fork([start_row + 0, start_row + 1])

    targetRadioMap.random_spoof(0, 100)
    targetRadioMap.random_spoof(1, 215668, completly_random=True)
    average_positions, errors = knn.run(
        0, 3, False, True, trainingRadioMap, targetRadioMap, log_penality=False)[:2]
    map = Map()

    map.plot_radioMap(trainingRadioMap,
                      title="Training data and test point", new_figure=True)
    map.plot_point(average_positions[0], args='ro')
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap, validationRadioMap.fork(
        [start_row]), log_penality=False)[0][0], args='yo')
    map.plot_point(targetRadioMap.get_positions()[0], args='go')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[0], confidence_value)

    map.plot_radioMap(trainingRadioMap,
                      title="Training data and test point", new_figure=True)
    map.plot_point(average_positions[1], args='ro')
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap, validationRadioMap.fork(
        [start_row + 1]), log_penality=False)[0][0], args='yo')
    map.plot_point(targetRadioMap.get_positions()[1], args='go')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[1], confidence_value)

    map.show()


def simu5_all_spoofing(seed: int):
    trainingRadioMap = RadioMap()
    trainingRadioMap.load('data/TrainingData.csv')
    validationRadioMap = RadioMap()
    validationRadioMap.load('data/ValidationData.csv')
    np.random.seed(seed)
    target_rows = np.random.randint(0, len(validationRadioMap.get_data()), 3)
    targetRadioMap = validationRadioMap.fork(target_rows)

    targetRadioMap.spoof(0, validationRadioMap, np.random.randint(
        0, len(validationRadioMap.get_data())))
    targetRadioMap.random_spoof(1, np.random.randint(0, 100000))
    targetRadioMap.random_spoof(
        2, np.random.randint(0, 100000), completly_random=True)

    average_positions, errors = knn.run(
        0, 3, False, True, trainingRadioMap, targetRadioMap, log_penality=False)[:2]

    map = Map()
    map.plot_radioMap(trainingRadioMap,
                      title="Spoofing with valid fingerprint (on row " + str(target_rows[0]) + ")", new_figure=True)
    map.plot_point(average_positions[0], args='ro', label='Estimated position')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[0], confidence_value)
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap,
                           validationRadioMap.fork([target_rows[0]]), log_penality=False)[0][0], args='yo', label='Estimated position without spoofing')
    map.plot_point(targetRadioMap.get_positions()[
                   0], args='go', label='Real position')

    map.plot_radioMap(trainingRadioMap,
                      title="Spoofing with alterate fingerprint (on row " + str(target_rows[1]) + ")", new_figure=True)
    map.plot_point(average_positions[1], args='ro', label='Estimated position')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[1], confidence_value)
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap,
                           validationRadioMap.fork([target_rows[1]]), log_penality=False)[0][0], args='yo', label='Estimated position without spoofing')
    map.plot_point(targetRadioMap.get_positions()[
                   1], args='go', label='Real position')

    map.plot_radioMap(trainingRadioMap,
                      title="Spoofing with completly random fingerprint (on row " + str(target_rows[2]) + ")", new_figure=True)
    map.plot_point(average_positions[2], args='ro', label='Estimated position')
    confidence_value = 16.12
    map.plot_confidence_circle(average_positions[2], confidence_value)
    map.plot_point(knn.run(0, 3, False, True, trainingRadioMap,
                           validationRadioMap.fork([target_rows[2]]), log_penality=False)[0][0], args='yo', label='Estimated position without spoofing')
    map.plot_point(targetRadioMap.get_positions()[
                   2], args='go', label='Real position')

    map.show()


if __name__ == '__main__':
    td = time.time()

    # simu1_find_best_k_neighbors()
    # simu2_spoofing()
    # simu3_find_error_mean()
    # simu4_randoom_spoofing()
    simu5_all_spoofing(5)

    print("Executed in ", time.time() - td, " seconds")
