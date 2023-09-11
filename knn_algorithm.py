""" KNN algorithm for indoor localization based on WAP fingerprints"""
import time
import numpy as np
from radio_map import RadioMap
from metadata_gen import load_ap_max
from fingerprint import Fingerprint


DURATION = 0


### Basic Positioning methods ###

def find_position_sc_method(k_neighbors: int, limit: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint) -> np.ndarray:
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
    if len(distances) < k_neighbors:
        return np.array([0, 0, 0])

    sorted_indexes = np.argsort(distances)
    average_position = np.mean(filtered_trning_pos_matrix[sorted_indexes[:k_neighbors]], axis=0)

    return average_position

def find_position_uc_method(k_neighbors: int, limit: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z).
    Retur (0,0,0) if the limit is too short for k_neighbors to be found."""

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
    if len(distances) < k_neighbors:
        return np.array([0, 0, 0])

    sorted_indexes = np.argsort(distances)
    average_position = np.mean(filtered_training_pos[sorted_indexes[:k_neighbors]], axis=0)

    return average_position

def find_position_vt_method(k_neighbors: int, limit_rate: float, trning_r_m: RadioMap, trgt_fgpt: Fingerprint) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z).
    Return (0,0,0) if the limit can't be found.
    VT is for Variable Threshlod."""

    trning_rss_matrix = trning_r_m.get_rss_matrix()
    trning_pos_matrix = trning_r_m.get_position_matrix()
    target_rss = trgt_fgpt.get_rss()

    unmatch_coord = np.sum(((target_rss == 100) & (trning_rss_matrix != 100)) | ((target_rss != 100) & (trning_rss_matrix == 100)), axis=1)
    match_coord = np.sum((target_rss != 100) & (trning_rss_matrix != 100) & (target_rss >= -85) & (trning_rss_matrix >= -85), axis=1)
    limit = limit_rate*np.sum((target_rss != 100) & (target_rss >= -85))
    filtered_diff_sq = np.square(target_rss - trning_rss_matrix[match_coord >= limit])
    filtered_trning_pos_matrix = trning_pos_matrix[match_coord >= limit]
    indexes_filtered_diff_sq = np.where(match_coord >= limit)[0]

    for filtered_diff_sq_i, diff_sq_i in enumerate(indexes_filtered_diff_sq):
        filtered_diff_sq[filtered_diff_sq_i][(target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100
                    ) | (target_rss == 100) & (trning_rss_matrix[diff_sq_i] != 100)] = 13*unmatch_coord[diff_sq_i]

    distances = np.sqrt(np.sum(filtered_diff_sq, axis=1))
    if len(distances) < k_neighbors:
        return np.array([0, 0, 0])

    sorted_indexes = np.argsort(distances)
    average_position = np.mean(filtered_trning_pos_matrix[sorted_indexes[:k_neighbors]], axis=0)

    return average_position


### Security filters ###

def secure_filter_overall_tolerance(trning_r_m: RadioMap, target_rss: np.ndarray, tolerance: float) -> bool:
    """ Return True if the fingerprint is valid, False otherwise."""

    aps_diameters, aps_centers = load_ap_max(trning_r_m)
    indexes_aps_to_check = np.where((target_rss != 100) & np.all(aps_centers != np.array([0, 0, 0]), axis=1))[0]

    n_ap = len(indexes_aps_to_check)

    failed_condition_tolerance = tolerance*((n_ap-1)*n_ap)/2
    failed_condition_counter = 0

    for i in range(len(indexes_aps_to_check)):
        for j in range(i+1, len(indexes_aps_to_check)):
            i_center = aps_centers[indexes_aps_to_check[i]]
            j_center = aps_centers[indexes_aps_to_check[j]]
            i_radius = aps_diameters[indexes_aps_to_check[i]]/2
            j_radius = aps_diameters[indexes_aps_to_check[j]]/2
            if np.linalg.norm(i_center - j_center) > i_radius + j_radius:
                failed_condition_counter += 1
                if failed_condition_counter > failed_condition_tolerance:
                    return False
    return True

def secure_filter_single_tolerance(trning_r_m: RadioMap, target_rss: np.ndarray, tolerance: float) -> bool:
    """ Return True if the fingerprint is valid, False otherwise."""

    aps_diameters, aps_centers = load_ap_max(trning_r_m)
    indexes_aps_to_check = np.where((target_rss != 100) & np.all(aps_centers != np.array([0, 0, 0]), axis=1))[0]

    n_ap = len(indexes_aps_to_check)

    failed_condition_tolerance = tolerance*(n_ap-1)
    failed_counter_matrix = np.zeros((n_ap, n_ap))

    for i in range(len(indexes_aps_to_check)):
        for j in range(i+1, len(indexes_aps_to_check)):
            i_center = aps_centers[indexes_aps_to_check[i]]
            j_center = aps_centers[indexes_aps_to_check[j]]
            i_radius = aps_diameters[indexes_aps_to_check[i]]/2
            j_radius = aps_diameters[indexes_aps_to_check[j]]/2
            if np.linalg.norm(i_center - j_center) > i_radius + j_radius:
                failed_counter_matrix[i][j] += 1
                failed_counter_matrix[j][i] += 1

    for i in range(n_ap):
        if np.sum(failed_counter_matrix[i]) > failed_condition_tolerance:
            return False

    return True


### KNN Algorythm ###

def find_position(k_neighbors: int, limit: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, method: str,filter_type: str = "", tolerance: float = 0.0) -> np.ndarray:
    """ Run the KNN algorithm. Return the estimated position (x, y, z).
    The method can be SC (Single Coordinate), UC (Unmatch Coordinate) or VT (Variable Threshold).
    The filter_type can be OF (Overall Filter) or PF (Pair Filter) or something else to disable the security filter."""

    target_rss = trgt_fgpt.get_rss()

    match filter_type:
        case "PF":
            valid_fingerprint = secure_filter_single_tolerance(trning_r_m, target_rss, tolerance)
        case "OF":
            valid_fingerprint = secure_filter_overall_tolerance(trning_r_m, target_rss, tolerance)
        case _:
            valid_fingerprint = True

    if valid_fingerprint:
        match method:
            case "SC":
                return find_position_sc_method(k_neighbors, limit, trning_r_m, trgt_fgpt)
            case "UC":
                return find_position_uc_method(k_neighbors, limit, trning_r_m, trgt_fgpt)
            case "VT":
                return find_position_vt_method(k_neighbors, limit, trning_r_m, trgt_fgpt)
    else:
        return np.array([0,0,0])


## Distance between predicted and actual position ##

def find_position_error(k_neighbors: int, limit: int, trning_r_m: RadioMap, trgt_fgpt: Fingerprint, method: str,filter_type: str = "", tolerance: float = 0.0, floor_height: float = 3.0) -> np.ndarray:
    """ Run the KNN secure algorithm. Return the distance between the estimated position and the actual position.
    Return None if the limit can't be found."""

    global DURATION
    td = time.time()
    predicted_position = find_position(k_neighbors, limit, trning_r_m, trgt_fgpt, method, filter_type, tolerance)
    DURATION = time.time() - td
    if (predicted_position == np.array([0,0,0])).all():
        return None
    actual_position = trgt_fgpt.get_position().copy()
    predicted_position[2] *= floor_height
    actual_position[2] *= floor_height
    return np.linalg.norm(predicted_position - actual_position)
