""" Load the blacklist for the knn_secure algo, if the filee deosn't existe create it"""
import os
import numpy as np

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

def load_ap_max(trning_r_m) -> list[int]:
    """ Load the ap_max from the file or generate it """
    if not os.path.isfile("data/ap_max_dist.txt"):
        with open("data/ap_max_dist.txt", "w") as file:
            all_rss = trning_r_m.get_rss()
            number_of_aps = len(all_rss[0])
            max_aps_distance = []

            for ap_index in range(number_of_aps):
                ap_rss_values = all_rss[:, ap_index]
                valid_rss_indexes = np.where(ap_rss_values != 100)[0]

                max_ap_distance = 0
                for i in range(len(valid_rss_indexes)):
                    for j in range(i + 1, len(valid_rss_indexes)):
                        distance = np.linalg.norm(trning_r_m.get_positions_by_row([valid_rss_indexes[i]]) - trning_r_m.get_positions_by_row([valid_rss_indexes[j]]))
                        max_ap_distance = max(max_ap_distance, distance)

                max_aps_distance.append(max_ap_distance)
                print(f"Max distance for AP {ap_index} is {max_ap_distance}")
            file.write(",".join(map(str, max_aps_distance)))
            return max_aps_distance
    else:
        with open("data/ap_max_dist.txt", "r") as file:
            ap_max = [float(ap) for ap in file.readline().strip().split(",")]
        return ap_max