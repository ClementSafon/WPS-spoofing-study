""" RadioMap class for loading and storing radio map data (fingerprints, positions, etc.)"""
import os
import copy
import csv
import numpy as np
from fingerprint import Fingerprint


class RadioMap:
    """ RadioMap class for loading and storing fingerprints."""

    def __init__(self) -> None:
        self.fingerprints = []
    
    def __str__(self) -> str:
        if len(self.fingerprints) > 2:
            return "RadioMap fingerprints : \n- " + str(self.fingerprints[0]) + "\n- ...\n- " + str(self.fingerprints[-1])
        elif len(self.fingerprints) == 2:
            return "RadioMap fingerprints : \n- " + str(self.fingerprints[0]) + "\n- " + str(self.fingerprints[-1])
        else:
            return "RadioMap fingerprint : \n- " + str(self.fingerprints[0])
    
    def __len__(self) -> int:
        return len(self.fingerprints)

    def get_fingerprint(self, id: int) -> Fingerprint:
        return self.fingerprints[id]

    def get_fingerprints(self) -> list[Fingerprint]:
        return self.fingerprints
    
    def add_fingerprint(self, fingerprint: Fingerprint) -> int:
        self.fingerprints.append(fingerprint)
        return len(self.fingerprints) - 1
    
    def remove_fingerprint(self, id: int) -> None:
        try:
            self.fingerprints = self.fingerprints[:id] + self.fingerprints[id + 1:]
        except IndexError:
            pass
    
    def get_rss_matrix(self) -> np.ndarray:
        return np.array([fgpt.rss for fgpt in self.fingerprints])
    
    def get_position_matrix(self) -> np.ndarray:
        return np.array([fgpt.position for fgpt in self.fingerprints])
    
    def get_position(self, id: int) -> np.ndarray:
        return self.fingerprints[id].position

    def load_from_csv(self, csv_file: str) -> None:
        if not os.path.isfile(csv_file):
            raise FileExistsError('File not found: ' + csv_file)
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fingerprint_id = 0
            for row in reader:
                new_fingerprint = Fingerprint(fingerprint_id, row)
                self.fingerprints.append(new_fingerprint)
                fingerprint_id += 1

    def fork(self, fingerprint_ids: list[int]) -> 'RadioMap':
        """ Create a new RadioMap object with a subset of the data."""
        new_map = RadioMap()
        if max(fingerprint_ids) >= len(self.fingerprints):
            raise ValueError('Row index out of range.')
        for fingerprint_id in fingerprint_ids:
            new_map.fingerprints.append(self.fingerprints[fingerprint_id].fork())
        return new_map
