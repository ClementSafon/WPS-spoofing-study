import csv
import numpy as np
import copy
import os


class RadioMap:

    def __init__(self) -> None:
        self.csv_file = None
        self.data = []

    def load(self, csv_file: str) -> None:
        if not os.path.isfile(csv_file):
            raise Exception('File not found: ' + csv_file)
        self.csv_file = csv_file
        with open(self.csv_file, 'r') as csvfile:
            try:
                reader = csv.DictReader(csvfile)
            except Exception as e:
                raise Exception('Error reading csv file: ' + str(e))
            for row in reader:
                data, rss = {}, []
                fields = reader.fieldnames
                if fields is None:
                    raise Exception('No fieldnames found in csv file.')
                for field in reader.fieldnames:
                    if field.startswith('WAP'):
                        rss.append(float(row[field]))
                    else:
                        data.update({field: float(row[field])})
                rss = np.array(rss)
                data.update({'rss': rss})
                self.data.append(data)

    def fork(self, rows: list[int]) -> 'RadioMap':
        new_map = RadioMap()
        new_map.csv_file = self.csv_file
        for row in rows:
            new_map.data.append(copy.deepcopy(self.data[row]))
        return new_map

    def spoof(self, spoofed_row: int, spoofing_radioMap: 'RadioMap', spoofing_row: int) -> None:
        for j in range(len(self.data[spoofed_row]['rss'])):
            if spoofing_radioMap.data[spoofing_row]['rss'][j] != 100:
                if self.data[spoofed_row]['rss'][j] != 100:
                    self.data[spoofed_row]['rss'][j] = max(
                        self.data[spoofed_row]['rss'][j], spoofing_radioMap.data[spoofing_row]['rss'][j])
                else:
                    self.data[spoofed_row]['rss'][j] = spoofing_radioMap.data[spoofing_row]['rss'][j]

    def random_spoof(self, spoofed_row: int, seed: int, completly_random=False) -> None:
        np.random.seed(seed)
        for j in range(len(self.data[spoofed_row]['rss'])):
            if completly_random:
                self.data[spoofed_row]['rss'][j] = np.random.randint(-100, -20)
                i = np.random.randint(0, 4)
                if i == 1 or i == 2 or i == 3:
                    self.data[spoofed_row]['rss'][j] = 100
            else:
                if self.data[spoofed_row]['rss'][j] != 100:
                    self.data[spoofed_row]['rss'][j] = np.random.randint(
                        -25, -20)
                    i = np.random.randint(0, 4)
                    if i == 1:
                        self.data[spoofed_row]['rss'][j] = 100

    def get_data(self) -> list:
        return self.data

    def get_data_by_row(self, start_row: int, stop_row: int) -> list[dict]:
        return self.data[start_row:stop_row]

    def print_data(self) -> None:
        for data in self.data:
            print('# ', data)

    def print_data_by_row(self, start_row: int, stop_row: int) -> None:
        i = 0
        for data in self.data[start_row:stop_row]:
            print("# [" + str(i + start_row) + "] " + str(data))
            i += 1

    def get_rss(self) -> np.array:
        rss = []
        for data in self.data:
            rss.append(np.array(data['rss']))
        return np.array(rss)

    def get_positions(self) -> np.array:
        pos = []
        for data in self.data:
            pos.append((data['LONGITUDE'], data['LATITUDE']))
        return np.array(pos)

    def get_floors(self) -> np.array:
        floors = []
        for data in self.data:
            floors.append(data['FLOOR'])
        return np.array(floors)
