""" KNN algorithm for indoor localization"""
import sys
import math
import numpy as np
from radio_map import RadioMap
import graph
from metadata_gen import load_blacklist, load_ap_max


def find_position(n_neighbors: int, trning_r_m: RadioMap, trgt_r_m: RadioMap, limit: int) -> tuple[int, int, int]:
    """ Run the KNN algorithm. Return the estimated position (x, y, z). 
    Retur (0,0,0) if the limit is too short for n_neighbors to be found."""

    training_rss = trning_r_m.get_rss()
    training_pos = trning_r_m.get_positions()

    if len(trgt_r_m.get_rss()) > 1:
        print("Error: target_rss should be a 1D array")
        sys.exit(1)
    
    target_rss = trgt_r_m.get_rss()[0]
    diff_sq = (target_rss - training_rss)**2
    
    for diff_sq_i in range(len(diff_sq)):
        match_coord = np.sum((target_rss != 100) & (training_rss[diff_sq_i] != 100))
        if match_coord < limit:
            diff_sq[diff_sq_i][:] = 1000000000
        else:
            diff_sq[diff_sq_i][(target_rss == 100) & (training_rss[diff_sq_i] != 100
                    ) | (target_rss == 100) & (training_rss[diff_sq_i] != 100)] = 0

    distances = np.sqrt(np.sum(diff_sq, axis=1))

    sorted_indices = np.argsort(distances)
    average_2d_position = np.mean(training_pos[sorted_indices[:n_neighbors]], axis=0)
    average_floor = np.mean(trning_r_m.get_floors()[sorted_indices[:n_neighbors]])

    if int(distances[sorted_indices[n_neighbors - 1]]) == 721110:
        estimated_position = np.array([0, 0, 0])
    else:
        estimated_position = np.array([average_2d_position[0], average_2d_position[1], average_floor])

    return estimated_position



def run_explicite(n_neighbors: int, trning_r_m=None, trgt_r_m=None, log_penality=True):
    """ Run the KNN algo, plot the k neighbors position, the average position and the true position of the test points."""
    distances = []
    training_rss = trning_r_m.get_rss()
    training_pos = trning_r_m.get_positions()
    average_positions, error_positions = [], []
    average_floors, error_floors = [], []

    for i_target_rss in range(len(trgt_r_m.get_rss())):
        target_rss = trgt_r_m.get_rss()[i_target_rss]
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

        print("Predicted position: ", average_position)
        print("Actual position: ", trgt_r_m.get_positions()[i_target_rss])
        print("Distance: ", error_position)

        average_positions.append(average_position)
        average_floors.append(average_floor)
        error_positions.append(error_position)
        error_floors.append(error_floor)

        graph.plot_radio_map(
            trning_r_m, title="Average location of the test point", new_figure=True)
        graph.plot_point(average_position, args='ro', label='Estimated position')
        graph.plot_point(trgt_r_m.get_positions()[i_target_rss], args='go', label='Actual position')
        for neigbors in training_pos[sorted_indices[:n_neighbors]]:
            graph.plot_point(neigbors, args='yo', label='K neighbors')
        graph.show()

    return (average_positions, error_positions, average_floors, error_floors)

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

def run(trning_r_m: RadioMap, trgt_r_m: RadioMap, n_neighbors: int, gui=True, verbose=True, limit=0) -> tuple[np.ndarray, float]:
    """ Run the KNN algorithm."""
    if verbose:
        print("Running KNN algorithm...")
    predicted_position = find_position(n_neighbors, trning_r_m, trgt_r_m, limit)
    actual_2d_position = trgt_r_m.get_positions()[0]
    actual_floor = trgt_r_m.get_floors()[0]
    floor_height = 3.0
    actual_position = np.array([actual_2d_position[0], actual_2d_position[1], actual_floor * floor_height])
    predicted_position = np.array([predicted_position[0], predicted_position[1], predicted_position[2] * floor_height])
    error_position = np.linalg.norm(
        predicted_position - actual_position)
    if verbose:
        print("Predicted position: ", predicted_position)
        print("Actual position: ", actual_position)
        print("Distance: ", error_position)
    if gui:
        display(trning_r_m, trgt_r_m, [predicted_position])
    return (predicted_position, error_position)

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

    r_m = RadioMap()
    r_m.load('data/TrainingData.csv')

    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')
    cli_args = parse_cli_arguments(arg_list)
    test_r_m = vld_r_m.fork([cli_args[1]])

    run(r_m, test_r_m, cli_args[0], cli_args[2], cli_args[3], cli_args[4])
