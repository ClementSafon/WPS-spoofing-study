""" Load the blacklist for the knn_secure algo, if the filee deosn't existe create it"""
import os
import numpy as np
from radio_map import RadioMap

def load_ap_max(trning_r_m: RadioMap) -> (np.ndarray, np.ndarray):
    """ Load the ap_max from the file or generate it """
    if not os.path.isfile("metadata/ap_max_dist.txt"):
        with open("metadata/ap_max_dist.txt", "w") as file:
            trning_rss_matrix = trning_r_m.get_rss_matrix()
            trning_pos_matrix = trning_r_m.get_position_matrix()
            number_of_aps = len(trning_rss_matrix[0])
            ap_max_data = []

            for ap_index in range(number_of_aps):
                ap_rssi_values = trning_rss_matrix[:, ap_index]
                valid_fingerprint_indexes = np.where(ap_rssi_values != 100)[0]

                #find max norm and center point
                max_ap_distance = 0
                positions = []
                rss_weights = []
                for i in range(len(valid_fingerprint_indexes)):
                    for j in range(i + 1, len(valid_fingerprint_indexes)):
                        distance = np.linalg.norm(trning_pos_matrix[valid_fingerprint_indexes[i]] - trning_pos_matrix[valid_fingerprint_indexes[j]])
                        max_ap_distance = max(max_ap_distance, distance)
                    positions.append(trning_pos_matrix[valid_fingerprint_indexes[i]])
                    rss_weights.append(1.0 / (np.abs(ap_rssi_values[valid_fingerprint_indexes[i]]) + 1))
                if len(positions) == 0:
                    center_point = np.array([0, 0, 0])
                else:
                    center_point = np.average(positions, axis=0, weights=rss_weights)
                
                ap_max_data.append((max_ap_distance, center_point))
                print(f"Max distance for AP {ap_index} is {max_ap_distance} centered at {center_point}")
            data_lines = []
            for max_dist, center_point in ap_max_data:
                data_lines.append(f"{max_dist},{center_point[0]},{center_point[1]},{center_point[2]}")
            file.write("\n".join(data_lines))
            return np.array([max_dist for max_dist, _ in ap_max_data]), np.array([center_point for _, center_point in ap_max_data])
    else:
        with open("metadata/ap_max_dist.txt", "r") as file:
            aps_max_diameters, aps_center_points = [], []
            for line in file:
                max_dist, x, y, z = line.strip().split(",")
                aps_max_diameters.append(float(max_dist))
                aps_center_points.append(np.array([float(x), float(y), float(z)]))
        return np.array(aps_max_diameters), np.array(aps_center_points)