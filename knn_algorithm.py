from RadioMap import *
from Map import Map
import sys
import random
import math
import matplotlib.pyplot as plt


def run(seed: int, n_neighbors: int, gui=True, verbose=True, trainingRadioMap=None, targetRadioMap=None, log_penality=True) -> list[tuple[np.ndarray, float]]:

    if trainingRadioMap is None:
        trainingRadioMap = RadioMap()
        trainingRadioMap.load('data/TrainingData.csv')

    if targetRadioMap is None:
        validationRadioMap = RadioMap()
        validationRadioMap.load('data/ValidationData.csv')
        id = seed
        targetRadioMap = validationRadioMap.fork(id, id + 1)
    distances = []
    trainingRss = trainingRadioMap.get_rss()
    trainingPos = trainingRadioMap.get_positions()
    average_positions = []
    error_positions = []
    average_floors = []
    error_floors = []
    c = 0
    for testRss in targetRadioMap.get_rss():
        if verbose:
            print('Computing distances...')
        diff_sq = (testRss - trainingRss)**2
        malus = 10
        for i in range(len(diff_sq)):
            n_unmatched_coord = np.sum((testRss == 100) & (
                trainingRss[i] != 100)) + np.sum((testRss != 100) & (trainingRss[i] == 100))
            if n_unmatched_coord != 0 and log_penality:
                diff_sq[i][(testRss == 100) & (trainingRss[i] != 100) | (testRss == 100) & (
                    trainingRss[i] != 100)] = 1/(math.log10(n_unmatched_coord*2)*malus)
            else:
                if n_unmatched_coord + np.sum((testRss == 100) & (
                        trainingRss[i] == 100)) == len(testRss):
                    diff_sq[i] = 1000000000
        distances = np.sqrt(np.sum(diff_sq, axis=1))

        sorted_indices = np.argsort(distances)
        average_position = np.mean(
            trainingPos[sorted_indices[:n_neighbors]], axis=0)
        average_floor = np.mean(trainingRadioMap.get_floors()[
                                sorted_indices[:n_neighbors]])
        error_position = np.linalg.norm(
            average_position - targetRadioMap.get_positions()[c])
        error_floor = abs(
            average_floor - targetRadioMap.get_floors()[c])

        if verbose:
            # print(distances, trainingPos[sorted_indices[:n_neighbors]])
            print("Predicted position: ", average_position)
            print("Actual position: ", targetRadioMap.get_positions()[c])
            print("Distance: ", error_position)

        average_positions.append(average_position)
        average_floors.append(average_floor)
        error_positions.append(error_position)
        error_floors.append(error_floor)
        c += 1

    if gui:
        map = Map()
        for i in range(len(average_positions)):
            map.plot_radioMap(
                trainingRadioMap, title="Training data and test point", new_figure=True)
            map.plot_point(average_positions[i], args='ro')
            map.plot_point(targetRadioMap.get_positions()[i], args='go')
        map.show()

    return (average_positions, error_positions, average_floors, error_floors)


if __name__ == '__main__':
    args = sys.argv
    if '-k' in args:
        n_neighbors = int(args[args.index('-k') + 1])
        args.pop(args.index('-k') + 1)
        args.pop(args.index('-k'))
    else:
        print('Usage: python3 knn_algorithm.py -k n_neighbors -s seed [--gui]')
        sys.exit(1)
    if '-s' in args:
        seed = int(args[args.index('-s') + 1])
        args.pop(args.index('-s') + 1)
        args.pop(args.index('-s'))
    else:
        print('Usage: python3 knn_algorithm.py -k n_neighbors -s seed [--gui]')
        sys.exit(1)
    if "--gui" in sys.argv:
        gui = True
        args.pop(args.index("--gui"))
    else:
        gui = False
    if "--verbose" in sys.argv:
        verbose = True
        args.pop(args.index("--verbose"))
    elif "-v" in sys.argv:
        verbose = True
        args.pop(args.index("-v"))
    else:
        verbose = False
    if len(args) > 1:
        print(
            'Usage: python3 knn_algorithm.py -k n_neighbors -s seed [--gui] [--verbose]')
        sys.exit(1)

    run(seed, n_neighbors, gui, verbose)
