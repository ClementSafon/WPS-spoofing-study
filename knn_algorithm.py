""" KNN algorithm for indoor localization"""
import sys
import math
import numpy as np
from radio_map import RadioMap
import graph


def run(n_neighbors: int, gui=True, verbose=True, trning_r_m=None, trgt_r_m=None, log_penality=True # noqa E501 pylint: disable=too-many-arguments, too-many-locals
        ) -> list[tuple[np.ndarray, float]]:
    """ Run the KNN algorithm."""
    distances = []
    training_rss = trning_r_m.get_rss()
    training_pos = trning_r_m.get_positions()
    average_positions, error_positions = [], []
    average_floors, error_floors = [], []

    for i_target_rss in range(len(trgt_r_m.get_rss())):
        target_rss = trgt_r_m.get_rss()[i_target_rss]
        if verbose:
            print('Computing distances...')
        diff_sq = (target_rss - training_rss)**2
        if log_penality:
            malus = 10
            for diff_sq_i in range(len(diff_sq)):
                n_unmatched_coord = np.sum((target_rss == 100) & (training_rss[diff_sq_i] != 100))
                n_unmatched_coord += np.sum((target_rss != 100) & (training_rss[diff_sq_i] == 100))
            if n_unmatched_coord != 0:
                penality = 1/(math.log10(n_unmatched_coord*2)*malus)
                diff_sq[diff_sq_i][(target_rss == 100) & (training_rss[diff_sq_i] != 100
                        ) | (target_rss == 100) & (training_rss[diff_sq_i] != 100)] = penality
        else:
            for training_rss_i, rss in enumerate(training_rss):
                if np.sum((target_rss != 100) & (rss != 100)) == 0:
                    diff_sq[training_rss_i] = 1000000000

        distances = np.sqrt(np.sum(diff_sq, axis=1))

        sorted_indices = np.argsort(distances)
        average_position = np.mean(
            training_pos[sorted_indices[:n_neighbors]], axis=0)
        average_floor = np.mean(trning_r_m.get_floors()[
                                sorted_indices[:n_neighbors]])
        error_position = np.linalg.norm(
            average_position - trgt_r_m.get_positions()[i_target_rss])
        error_floor = abs(
            average_floor - trgt_r_m.get_floors()[i_target_rss])

        if verbose:
            # print(distances, training_pos[sorted_indices[:n_neighbors]])
            print("Predicted position: ", average_position)
            print("Actual position: ", trgt_r_m.get_positions()[i_target_rss])
            print("Distance: ", error_position)

        average_positions.append(average_position)
        average_floors.append(average_floor)
        error_positions.append(error_position)
        error_floors.append(error_floor)

    if gui:
        display(trning_r_m, trgt_r_m, average_positions)

    return (average_positions, error_positions, average_floors, error_floors)

def display(trning_r_m, trgt_r_m, average_positions):
    """ Display the results."""
    for i, average_position in enumerate(average_positions):
        graph.plot_radio_map(
            trning_r_m, title="Average location of the test point", new_figure=True)
        graph.plot_point(average_position, args='ro')
        graph.plot_point(trgt_r_m.get_positions()[i], args='go')
    graph.show()

def print_usage() -> None:
    """ Print the usage of the script."""
    print('Usage: python3 knn_algorithm.py -k <k_neigbors> -r <rows, ex: 1,2,5> [--gui] [--verbose]') # noqa E501 pylint: disable=line-too-long

def parse_cli_arguments(args: list[str]) -> tuple[int, list[int], bool, bool]:
    """ Parse the CLI arguments."""
    k_neighbors, rows_from_validation_dataset = 0, []
    if '-k' in args:
        try:
            k_neighbors = int(args[args.index('-k') + 1])
        except ValueError:
            print_usage()
            sys.exit(1)
        args.pop(args.index('-k') + 1)
        args.pop(args.index('-k'))
    else:
        print_usage()
        sys.exit(1)
    if '-r' in args:
        rows_from_validation_dataset = args[args.index('-r') + 1].split(',')
        for rows_i, row in enumerate(rows_from_validation_dataset):
            try:
                rows_from_validation_dataset[rows_i] = int(row)
            except ValueError:
                print_usage()
                sys.exit(1)
        args.pop(args.index('-r') + 1)
        args.pop(args.index('-r'))
    else:
        print_usage()
        sys.exit(1)
    if "--gui" in args:
        gui = True
        args.pop(args.index("--gui"))
    else:
        gui = False
    if "--verbose" in args:
        verbose = True
        args.pop(args.index("--verbose"))
    if len(args) >= 1:
        print_usage()
        sys.exit(1)
    return k_neighbors, rows_from_validation_dataset, gui, verbose

if __name__ == '__main__':
    arg_list = sys.argv

    r_m = RadioMap()
    r_m.load('data/TrainingData.csv')

    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')
    test_r_m = vld_r_m.fork(parse_cli_arguments(arg_list)[1])

    run([parse_cli_arguments(arg_list)[i] for i in (0,2,3)], r_m, test_r_m)
