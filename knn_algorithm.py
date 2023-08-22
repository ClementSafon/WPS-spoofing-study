""" KNN algorithm for indoor localization based on WAP fingerprints"""
import sys
import math
import numpy as np
from radio_map import RadioMap
import graph
from metadata_gen import load_blacklist, load_ap_max
from fingerprint import Fingerprint
import time

def find_position_SC_method(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int) -> np.ndarray:
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
                    ) | (target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100)] = 13*unmatch_coord[diff_sq_i]

    distances = np.sqrt(np.sum(filtered_diff_sq, axis=1))
    if len(distances) < n_neighbors:
        return np.array([0, 0, 0])

    sorted_indices = np.argsort(distances)
    average_position = np.mean(filtered_training_pos[sorted_indices[:n_neighbors]], axis=0)

    return average_position

def find_position_VT_method(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit_rate: float) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z). 
    Return (0,0,0) if the limit can't be found.
    VT is for Variable Threshlod."""

    trning_rss_matrix = trning_r_m.get_rss_matrix()
    trning_pos_matrix = trning_r_m.get_position_matrix()
    target_rss = trgt_fgpt.get_rss()

    match_coord = np.sum((target_rss != 100) & (trning_rss_matrix != 100) & (target_rss >= -85) & (trning_rss_matrix >= -85), axis=1)
    limit = limit_rate*np.sum((target_rss != 100) & (target_rss >= -85))
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

def find_position_error(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int, floor_height = 3.0, method="SC") -> np.ndarray:
    match method:
        case "SC":
            predicted_position = find_position_SC_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case "UC":
            predicted_position = find_position_UC_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case "VT":
            predicted_position = find_position_VT_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case _:
            print("Invalid method")
            return np.inf
    if (predicted_position == np.array([0,0,0])).all():
        return np.inf
    actual_position = trgt_fgpt.get_position().copy()
    predicted_position[2] *= floor_height
    actual_position[2] *= floor_height
    return np.linalg.norm(predicted_position - actual_position)  

def find_position_error_article(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int, method="SC") -> np.ndarray:
    match method:
        case "SC":
            predicted_position = find_position_SC_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case "UC":
            predicted_position = find_position_UC_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
        case _:
            print("Invalid method")
            return np.inf
    if (predicted_position == np.array([0,0,0])).all():
        return np.inf
    actual_position = trgt_fgpt.get_position().copy()
    if int(actual_position[2]) != int(predicted_position[2]):
        return np.inf
    return np.linalg.norm(predicted_position[:2] - actual_position[:2])

# Security

def find_position_secure(n_neighbors: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, limit: int) -> np.ndarray:
    """ Run the KNN secure algorithm."""

    trning_pos_matrix = trning_r_m.get_position_matrix()
    target_rss = trgt_fgpt.get_rss()

    valid_fingerprint = True

    aps_max = load_ap_max(trning_r_m)
    for rssi_index, rssi in enumerate(target_rss):
        if rssi != 100:
            for rssi_index_comp, rssi_comp in enumerate(target_rss):
                if rssi_comp != 100:
                    max_dist = max(aps_max[rssi_index][0]/2 + aps_max[rssi_index_comp][0]/2, 60)
                    center_point = aps_max[rssi_index][1]
                    center_point_comp = aps_max[rssi_index_comp][1]
                    if (center_point != np.array([0,0,0])).all() and (center_point_comp != np.array([0,0,0])).all():
                        if np.linalg.norm(center_point - center_point_comp) > max_dist:
                            valid_fingerprint = False
                            break

    if valid_fingerprint:
        return find_position_UC_method(n_neighbors, trning_r_m, trgt_fgpt, limit)
    else:
        return np.array([0,0,0])


## Section used to run by itself

def run(trning_r_m: RadioMap, test_fgpt: Fingerprint, n_neighbors: int, gui=True, verbose=True, limit=0) -> tuple[np.ndarray, float]:
    """ Run the KNN algorithm."""
    if verbose:
        print("Running KNN algorithm...")
    predicted_position = find_position_SC_method(n_neighbors, trning_r_m, test_fgpt, limit)
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

