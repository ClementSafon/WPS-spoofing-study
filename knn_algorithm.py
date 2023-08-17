""" KNN algorithm for indoor localization based on WAP fingerprints"""
import sys
import math
import numpy as np
from radio_map import RadioMap
import graph
from metadata_gen import load_blacklist, load_ap_max
from fingerprint import Fingerprint
import time

def find_position(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z). 
    Return (0,0,0) if the limit can't be found."""

    trning_rss_matrix = trning_r_m.get_rss_matrix()
    trning_pos_matrix = trning_r_m.get_position_matrix()
    target_rss = trgt_fgpt.get_rss()

    match_coord = np.sum((target_rss != 100) & (trning_rss_matrix != 100), axis=1)
    filtered_diff_sq = np.square(target_rss - trning_rss_matrix[match_coord >= limit])
    filtered_trning_pos_matrix = trning_pos_matrix[match_coord >= limit]
    indexes_filtered_diff_sq = np.where(match_coord >= limit)[0]
    
    for filtered_diff_sq_i, diff_sq_i in enumerate(indexes_filtered_diff_sq):
        filtered_diff_sq[filtered_diff_sq_i][(target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100
                    ) | (target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100)] = 0

    distances = np.sqrt(np.sum(filtered_diff_sq, axis=1))
    if len(distances) < n_neighbors:
        return np.array([0, 0, 0])

    sorted_indices = np.argsort(distances)
    average_position = np.mean(filtered_trning_pos_matrix[sorted_indices[:n_neighbors]], axis=0)

    return average_position

def find_position_UC_method(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z). 
    Retur (0,0,0) if the limit is too short for n_neighbors to be found."""

    trning_rss_matrix = trning_r_m.get_rss_matrix()
    trning_pos_matrix = trning_r_m.get_position_matrix()
    target_rss = trgt_fgpt.get_rss()

    unmatch_coord = np.sum(((target_rss == 100) & (trning_rss_matrix != 100)) | ((target_rss != 100) & (trning_rss_matrix == 100)), axis=1)
    filtered_diff_sq = np.square(target_rss - trning_rss_matrix[unmatch_coord < limit])
    indexes_filtered_diff_sq = np.where(unmatch_coord < limit)[0]
    filtered_training_pos = trning_pos_matrix[unmatch_coord < limit]
    
    for filtered_diff_sq_i, diff_sq_i in enumerate(indexes_filtered_diff_sq):
        filtered_diff_sq[filtered_diff_sq_i][(target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100
                    ) | (target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100)] = 0

    distances = np.sqrt(np.sum(filtered_diff_sq, axis=1))
    if len(distances) < n_neighbors:
        return np.array([0, 0, 0])

    sorted_indices = np.argsort(distances)
    average_position = np.mean(filtered_training_pos[sorted_indices[:n_neighbors]], axis=0)

    return average_position

def find_position_error(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int, floor_height = 3.0, method="SC") -> np.ndarray:
    match method:
        case "SC":
            predicted_position = find_position(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case "UC":
            predicted_position = find_position_UC_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case "OT":
            predicted_position = find_position_other_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case _:
            print("Invalid method")
            return np.inf
    if (predicted_position == np.array([0,0,0])).all():
        return np.inf
    actual_position = trgt_fgpt.get_position().copy()
    predicted_position[2] *= floor_height
    actual_position[2] *= floor_height
    return np.linalg.norm(predicted_position - actual_position)

def find_position_other_method(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z). 
    Retur (0,0,0) if the limit is too short for n_neighbors to be found."""

    trning_rss_matrix = trning_r_m.get_rss_matrix()
    trning_pos_matrix = trning_r_m.get_position_matrix()
    target_rss = trgt_fgpt.get_rss()

    unmatch_coord = np.sum(((target_rss == 100) & (trning_rss_matrix != 100)) | ((target_rss != 100) & (trning_rss_matrix == 100)), axis=1)
    filtered_diff_sq = np.square(target_rss - trning_rss_matrix[unmatch_coord < limit])
    indexes_filtered_diff_sq = np.where(unmatch_coord < limit)[0]
    filtered_training_pos = trning_pos_matrix[unmatch_coord < limit]
    
    for filtered_diff_sq_i, diff_sq_i in enumerate(indexes_filtered_diff_sq):
        filtered_diff_sq[filtered_diff_sq_i][(target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100
                    ) | (target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100)] = 13*unmatch_coord[diff_sq_i]

    distances = np.sqrt(np.sum(filtered_diff_sq, axis=1))
    if len(distances) < n_neighbors:
        return np.array([0, 0, 0])

    sorted_indices = np.argsort(distances)
    average_position = np.mean(filtered_training_pos[sorted_indices[:n_neighbors]], axis=0)

    return average_position


def run_secure(n_neighbors: int, verbose=True, trning_r_m=None, trgt_r_m=None, method="neigbors") -> list[tuple[np.ndarray, float]]:
    """ Run the KNN secure algorithm."""

    if method == "neigbors":
        aps_blacklist = load_blacklist(trning_r_m)    

        def check_attack() -> bool:
            for rss_index, rss in enumerate(target_rss):
                if rss != 100:
                    black_list = aps_blacklist[rss_index]
                    for rss_index_comp, rss_comp in enumerate(target_rss):
                        if rss_index_comp in black_list and rss_comp != 100:
                            return False
            return True
    elif method == "distance":
        aps_max = load_ap_max(trning_r_m)

        def check_attack() -> bool:
            for rss_index, rss in enumerate(target_rss):
                if rss != 100:
                    for rss_index_comp, rss_comp in enumerate(target_rss):
                        if rss_comp != 100 and np.linalg.norm(trning_r_m.get_positions()[rss_index] - trning_r_m.get_positions()[rss_index_comp]) > aps_max[rss_index]:
                            print(rss_index, rss_index_comp)
                            return False
            return True
    elif method == "distance_OMAX":
        # aps_max = load_ap_max(trning_r_m)
        # oa_max = max(aps_max)
        oa_max = 404.0

        def check_attack() -> bool:
            max_distance_threshold = 404.0
            target_positions = trning_r_m.get_positions()
            target_rss_np = np.array(target_rss)

            for rss_index, rss in enumerate(target_rss_np):
                if rss != 100:
                    valid_rss_index = np.where(target_rss_np != 100)[0]
                    distances = np.linalg.norm(target_positions[rss_index] - target_positions[valid_rss_index], axis=1)
                    if np.any(distances > max_distance_threshold):
                        return False
            return True
    elif method == "distance_FA":
        aps_max = load_ap_max(trning_r_m)

        def check_attack() -> bool:
            for rss_index, rss in enumerate(target_rss):
                if rss != 100:
                    for rss_index_comp, rss_comp in enumerate(target_rss):
                        if rss_comp != 100 and np.linalg.norm(trning_r_m.get_positions()[rss_index] - trning_r_m.get_positions()[rss_index_comp]) > aps_max[rss_index] + 200:
                            return False
            return True


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
        for training_rss_i, rss in enumerate(training_rss):
            if np.sum((target_rss != 100) & (rss != 100)) == 0:
                diff_sq[training_rss_i] = 1000000000

        distances = np.sqrt(np.sum(diff_sq, axis=1))

        if not check_attack():
            average_position = [-1, -1]
            average_floor = -1
            error_position = -1
            error_floor = -1
        else:
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
            print("Predicted position: ", average_position)
            print("Actual position: ", trgt_r_m.get_positions()[i_target_rss])
            print("Distance: ", error_position)

        average_positions.append(average_position)
        average_floors.append(average_floor)
        error_positions.append(error_position)
        error_floors.append(error_floor)

    return (average_positions, error_positions, average_floors, error_floors)


## Section used to run by itself

def run(trning_r_m: RadioMap, test_fgpt: Fingerprint, n_neighbors: int, gui=True, verbose=True, limit=0) -> tuple[np.ndarray, float]:
    """ Run the KNN algorithm."""
    if verbose:
        print("Running KNN algorithm...")
    predicted_position = find_position(n_neighbors, trning_r_m, test_fgpt, limit)
    actual_position = test_fgpt.get_position().copy()
    floor_height = 3.0
    predicted_position[2] *= floor_height
    actual_position[2] *= floor_height
    error_position = np.linalg.norm(
        predicted_position - actual_position)
    if verbose:
        print("Predicted position: ", predicted_position)
        print("Actual position: ", actual_position)
        print("Distance: ", error_position)
    if gui:
        graph.plot_radio_map(
            trning_r_m, title="Average location of the test point", new_figure=True)
        graph.plot_point(predicted_position, args='ro', label="Predicted Position")
        graph.plot_point(test_fgpt.get_position(), args='go', label='Real Position')
        graph.show()
    return (predicted_position, error_position)

def print_usage() -> None:
    """ Print the usage of the script."""
    print('Usage: python3 knn_algorithm.py -k <k_neigbors> -r <row> -l <limit> [--gui] [--verbose]') # noqa E501 pylint: disable=line-too-long

def parse_cli_arguments(args: list[str]) -> tuple[int, int, bool, bool]:
    """ Parse the CLI arguments."""
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
        row_from_validation_dataset = int(args[args.index('-r') + 1])
        args.pop(args.index('-r') + 1)
        args.pop(args.index('-r'))
    else:
        print_usage()
        sys.exit(1)
    if '-l' in args:
        try:
            limit = int(args[args.index('-l') + 1])
        except ValueError:
            print_usage()
            sys.exit(1)
        args.pop(args.index('-l') + 1)
        args.pop(args.index('-l'))
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
    return k_neighbors, row_from_validation_dataset, gui, verbose, limit

if __name__ == '__main__':
    arg_list = sys.argv[1:]
    cli_args = parse_cli_arguments(arg_list)

    print("loading data...")
    r_m = RadioMap()
    r_m.load_from_csv('data/TrainingData.csv')

    vld_r_m = RadioMap()
    vld_r_m.load_from_csv('data/ValidationData.csv')
    print("Done !")
    
    test_fgpt = vld_r_m.get_fingerprint(cli_args[1])

    run(r_m, test_fgpt, cli_args[0], cli_args[2], cli_args[3], cli_args[4])
    run(r_m, test_fgpt, cli_args[0], cli_args[2], cli_args[3], cli_args[4])

