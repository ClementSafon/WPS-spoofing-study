""" Load the blacklist for the knn_secure algo, if the filee deosn't existe create it"""
import os
import numpy as np
from radio_map import RadioMap

def load_blacklist(trning_r_m) -> list[list]:
    """ Load the blacklist from the file """
    if not os.path.isfile("data/blacklist.txt"):
        with open("data/blacklist.txt", "w") as file:
            aps_blacklist = []
            all_rss = trning_r_m.get_rss()
            number_of_aps = len(all_rss[0])
            for ap_index in range(number_of_aps):
                print(f"Generating blacklist for AP {ap_index}")
                ap_blacklist = [i for i in range(number_of_aps)]
                for rss_index, rss in enumerate(all_rss):
                    if rss[ap_index] != 100:
                        for blacklisted_ap in ap_blacklist:
                            if all_rss[rss_index][blacklisted_ap] != 100:
                                ap_blacklist.remove(blacklisted_ap)
                aps_blacklist.append(ap_blacklist)
            file.write("\n".join([",".join([str(ap_index) for ap_index in ap_blacklist]) for ap_blacklist in aps_blacklist]))
            return aps_blacklist
    else:
        aps_blacklist = []
        with open("data/blacklist.txt", "r") as file:
            for line in file:
                ap_blacklist = [int(ap) for ap in line.strip().split(",")]
                aps_blacklist.append(ap_blacklist)
        return aps_blacklist

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
                for i in range(len(valid_fingerprint_indexes)):
                    for j in range(i + 1, len(valid_fingerprint_indexes)):
                        distance = np.linalg.norm(trning_pos_matrix[valid_fingerprint_indexes[i]] - trning_pos_matrix[valid_fingerprint_indexes[j]])
                        max_ap_distance = max(max_ap_distance, distance)
                    positions.append(trning_pos_matrix[valid_fingerprint_indexes[i]])
                if len(positions) == 0:
                    center_point = np.array([0, 0, 0])
                else:
                    center_point = np.mean(positions, axis=0)
                
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