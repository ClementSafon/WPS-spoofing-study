from radio_map import RadioMap
import numpy as np
import copy
import csv

def create_dataset(radio_map: RadioMap, output_file: str):
    """
    Create a dataset from a radio map.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        column_names = []
        for i in range(1, len(radio_map.get_data()[0]['rss']) + 1):
            column_names.append('WAP' + "0"*(len(str(520))-len(str(i))) + str(i))
        column_names += ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']
        writer.writerow(column_names)
        for fingerprint in radio_map.get_data():
            row_data = []
            for rss in fingerprint['rss']:
                row_data.append(str(rss))
            row_data += [fingerprint['LONGITUDE'], fingerprint['LATITUDE'], fingerprint['FLOOR'], fingerprint['BUILDINGID'], fingerprint['SPACEID'], fingerprint['RELATIVEPOSITION'], fingerprint['USERID'], fingerprint['PHONEID'], fingerprint['TIMESTAMP']]
            writer.writerow(row_data)
    print("Dataset created in " + output_file)


def create_spoofed_r_m(radio_map: RadioMap, n_spoofed_AP: int, min_AP_on_rss: int) -> RadioMap:
    """
    Create a radio map victim of AP spoofing from a radio map.
    """
    victim_radio_map = RadioMap()
    victim_radio_map.csv_file = radio_map.csv_file
    for fingerprint in radio_map.get_data():
        new_fingerprint = copy.deepcopy(fingerprint)
        if np.sum(np.array(fingerprint['rss']) != 100) >= min_AP_on_rss:
            row = np.random.randint(0, len(radio_map.get_data()))
            while fingerprint["BUILDINGID"] != radio_map.get_data()[row]["BUILDINGID"]:
                row = np.random.randint(0, len(radio_map.get_data()))
            false_fingerprint = radio_map.get_data()[row]['rss']
            indexes = np.nonzero(np.array(false_fingerprint) - 100)[0]
            spoofed_AP_indexes = np.random.choice(indexes, n_spoofed_AP)
            for index in spoofed_AP_indexes:
                if fingerprint['rss'][index] != 100:
                    new_fingerprint['rss'][index] = max(false_fingerprint[index], fingerprint['rss'][index])
                else:
                    new_fingerprint['rss'][index] = false_fingerprint[index]
        victim_radio_map.data.append(new_fingerprint)
    return victim_radio_map


if __name__ == '__main__':
    radio_map = RadioMap()
    radio_map.load('data/ValidationData.csv')

    for i in range(1, 11):
        victim_r_m = create_spoofed_r_m(radio_map, i, 3)
        create_dataset(victim_r_m, 'clement_data/ValidationData_' + str(i) + '.csv')