""" Create a corrupted dataset (Scenario 2). """
import copy
import csv
import numpy as np
from radio_map import RadioMap

def create_dataset(radio_map: RadioMap, output_file: str):
    """
    Create a dataset from a radio map.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        column_names = []
        for i in range(1, 520 + 1):
            column_names.append('WAP' + "0"*(len(str(520))-len(str(i))) + str(i))
        column_names += ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']
        writer.writerow(column_names)
        for fingerprint in radio_map.get_fingerprints():
            row_data = []
            for rss in fingerprint.get_rss():
                row_data.append(str(rss))
            row_data += [fingerprint.get_position()[0], fingerprint.get_position()[1], fingerprint.get_position()[2],
                         fingerprint.get_building_id(), fingerprint.get_space_id(), fingerprint.get_relative_position(),
                         fingerprint.get_user_id(), fingerprint.get_phone_id(), fingerprint.get_timestamp()]
            writer.writerow(row_data)
    print("Dataset created in " + output_file)


def create_spoofed_r_m(radio_map: RadioMap, n_spoofed_ap: int, min_ap_on_rss: int) -> RadioMap:
    """
    Create a radio map victim of AP spoofing from a radio map. Scenario 2
    """
    victim_radio_map = RadioMap()
    for fingerprint in radio_map.get_fingerprints():
        new_fingerprint = fingerprint.fork()
        if np.sum(np.array(fingerprint.get_rss()) != 100) >= min_ap_on_rss:
            row = np.random.randint(0, len(radio_map))
            false_fingerprint_rss_vector = radio_map.get_fingerprint(row).get_rss()
            indexes = np.where(false_fingerprint_rss_vector != 100)[0]
            while fingerprint.get_building_id == radio_map.get_fingerprint(row).get_building_id() or len(indexes) < n_spoofed_ap:
                row = np.random.randint(0, len(radio_map))
                false_fingerprint_rss_vector = radio_map.get_fingerprint(row).get_rss()
                indexes = np.where(false_fingerprint_rss_vector != 100)[0]
            spoofed_ap_indexes = np.random.choice(indexes, n_spoofed_ap)
            new_fingerprint_rss = copy.deepcopy(new_fingerprint.get_rss())
            for index in spoofed_ap_indexes:
                new_rss_value = new_fingerprint_rss[index]
                if new_rss_value != 100:
                    new_rss_value = max(false_fingerprint_rss_vector[index], new_rss_value)
                else:
                    new_rss_value = false_fingerprint_rss_vector[index]
                new_fingerprint_rss[index] = new_rss_value
            new_fingerprint.update_rss(new_fingerprint_rss)
        victim_radio_map.add_fingerprint(new_fingerprint)
    return victim_radio_map


if __name__ == '__main__':
    radio_map = RadioMap()
    radio_map.load_from_csv('datasets/ValidationData.csv')

    for n in range(1, 11):
        create_dataset(create_spoofed_r_m(radio_map, n, 1), 'datasets/corrupted/scenario2/ValidationData_' + str(n) + '.csv')
