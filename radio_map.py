""" RadioMap class for loading and storing radio map data (fingerprints, positions, etc.)"""
import os
import copy
import csv
import numpy as np



class RadioMap:
    """ RadioMap class for loading and storing radio map data (fingerprints, positions, etc.)"""

    def __init__(self) -> None:
        self.csv_file = None
        self.data = []

    def load(self, csv_file: str) -> None:
        """ Load the radio map data from a csv file."""
        if not os.path.isfile(csv_file):
            raise FileExistsError('File not found: ' + csv_file)
        self.csv_file = csv_file
        with open(self.csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                exepected_fields = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP'] # noqa: E501 pylint: disable=line-too-long
                if not all(field in reader.fieldnames for field in exepected_fields):
                    raise ValueError('Invalid csv file.')
                data = {'LONGITUDE': float(row['LONGITUDE']), 'LATITUDE': float(row['LATITUDE']), 'FLOOR': int(row['FLOOR']), 'BUILDINGID': int(row['BUILDINGID']), 'SPACEID': int( # noqa: E501 pylint: disable=line-too-long
                    row['SPACEID']), 'RELATIVEPOSITION': int(row['RELATIVEPOSITION']), 'USERID': int(row['USERID']), 'PHONEID': int(row['PHONEID']), 'TIMESTAMP': row['TIMESTAMP'], 'rss': []} # noqa: E501 pylint: disable=line-too-long
                for field in reader.fieldnames:
                    if field.startswith('WAP'):
                        data['rss'].append(int(row[field]))
                data['rss'] = np.array(data['rss'])
                self.data.append(data)

    def fork(self, rows: list[int]) -> 'RadioMap':
        """ Create a new RadioMap object with a subset of the data."""
        new_map = RadioMap()
        new_map.csv_file = self.csv_file
        if max(rows) >= len(self.data):
            raise ValueError('Row index out of range.')
        for row in rows:
            new_map.data.append(copy.deepcopy(self.data[row]))
        return new_map

    def spoof(self, spoofed_row: int, spoofing_radio_map: 'RadioMap', spoofing_row: int) -> None:
        """ Spoof the data of a row with the data of another row from another radio map."""
        if spoofed_row >= len(self.data) or spoofing_row >= len(spoofing_radio_map.get_data()):
            raise ValueError('Row index out of range.')
        for j in range(len(self.data[spoofed_row]['rss'])):
            spoofing_value = spoofing_radio_map.data[spoofing_row]['rss'][j]
            if spoofing_value != 100:
                spoofing_value = np.random.normal(spoofing_radio_map.data[spoofing_row]['rss'][j], 2.26)
                if self.data[spoofed_row]['rss'][j] != 100:
                    self.data[spoofed_row]['rss'][j] = max(
                        self.data[spoofed_row]['rss'][j], spoofing_value)
                else:
                    self.data[spoofed_row]['rss'][j] = spoofing_value

    def reemitting_spoof(self, spoofed_row: int) -> None:
        """ Spoof the APs by reemitting the beacon full power."""
        if spoofed_row >= len(self.data):
            raise ValueError('Row index out of range.')
        for j in range(len(self.data[spoofed_row]['rss'])):
            if self.data[spoofed_row]['rss'][j] != 100:
                self.data[spoofed_row]['rss'][j] = np.random.normal(-38, 2.26)

    def random_spoof(self, spoofed_row: int, seed: int) -> None:
        """ Spoof the data of a row with random data."""
        if spoofed_row >= len(self.data):
            raise ValueError('Row index out of range.')
        np.random.seed(seed)
        for j in np.random.randint(0, len(self.data[spoofed_row]['rss']), np.random.randint(1, 10)):
            self.data[spoofed_row]['rss'][j] = np.random.randint(-87, -35)

    def get_data(self) -> list[dict]:
        """ Return the data."""
        return self.data

    def get_data_by_row(self, rows) -> list[dict]:
        """ Return the data from a subset of the rows."""
        if max(rows) >= len(self.data):
            raise ValueError('Row index out of range.')
        data = []
        for row in rows:
            data.append(self.data[row])
        return data

    def print_data(self) -> None:
        """ Print the data."""
        for data in self.data:
            print('##################')
            for key, value in data.items():
                print('# ', key, ': ', value)

    def print_data_by_row(self, rows) -> None:
        """ Print the data from a subset of the rows."""
        if max(rows) >= len(self.data):
            raise ValueError('Row index out of range.')
        for row in rows:
            print('##################')
            for key, value in self.data[row].items():
                print('# ', key, ': ', value)

    def get_rss(self) -> np.array:
        """ Return the RSS values."""
        rss = []
        for data in self.data:
            rss.append(np.array(data['rss']))
        return np.array(rss)

    def get_positions(self) -> np.array:
        """ Return the positions (longitude, latitude)."""
        pos = []
        for data in self.data:
            pos.append((data['LONGITUDE'], data['LATITUDE']))
        return np.array(pos)

    def get_positions_by_row(self, rows: list[int]) -> np.array:
        """ Return the positions (longitude, latitude) from a subset of the rows."""
        if max(rows) >= len(self.data):
            raise ValueError('Row index out of range.')
        pos = []
        for row in rows:
            pos.append((self.data[row]['LONGITUDE'], self.data[row]['LATITUDE']))
        return np.array(pos)

    def get_floors(self) -> np.array:
        """ Return the floor numbers."""
        floors = []
        for data in self.data:
            floors.append(data['FLOOR'])
        return np.array(floors)
