""" RadioMap class for loading and storing radio map data (fingerprints, positions, etc.)"""
import os
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
        if len(self.fingerprints) == 2:
            return "RadioMap fingerprints : \n- " + str(self.fingerprints[0]) + "\n- " + str(self.fingerprints[-1])
        return "RadioMap fingerprint : \n- " + str(self.fingerprints[0])

    def __len__(self) -> int:
        return len(self.fingerprints)

    def get_fingerprint(self, id_: int) -> Fingerprint:
        """ Get the fingerprint with the given ID."""
        return self.fingerprints[id_]

    def get_fingerprints(self) -> list[Fingerprint]:
        """ Get the fingerprints."""
        return self.fingerprints

    def add_fingerprint(self, fingerprint: Fingerprint) -> int:
        """ Add a fingerprint to the radio map."""
        self.fingerprints.append(fingerprint)
        return len(self.fingerprints) - 1

    def remove_fingerprint(self, id_: int) -> None:
        """ Remove a fingerprint from the radio map."""
        try:
            self.fingerprints = self.fingerprints[:id_] + self.fingerprints[id_ + 1:]
        except IndexError:
            pass

    def get_rss_matrix(self) -> np.ndarray:
        """ Get the RSS matrix."""
        return np.array([fgpt.get_rss() for fgpt in self.fingerprints])

    def get_position_matrix(self) -> np.ndarray:
        """ Get the position matrix."""
        return np.array([fgpt.get_position() for fgpt in self.fingerprints])

    def load_from_csv(self, csv_file: str) -> None:
        """  Load a radio map from a CSV file."""
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
