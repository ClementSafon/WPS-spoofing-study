""" This module defines the Fingerprint class."""
import numpy as np

class Fingerprint():
    """ Class to represent a fingerprint."""

    def __init__(self, id_: int, csv_row: dict) -> int:
        """ Initialize the fingerprint."""
        try:
            self.id_ = id_
            self.timestamp = int(csv_row['TIMESTAMP'])
            self.user_id = int(csv_row['USERID'])
            self.phone_id = int(csv_row['PHONEID'])
            self.space_id = int(csv_row['SPACEID'])
            self.relative_position = int(csv_row['RELATIVEPOSITION'])
            self.building_id = int(csv_row['BUILDINGID'])
            self.position = np.array([float(csv_row['LONGITUDE']), float(csv_row['LATITUDE']), float(csv_row['FLOOR'])])
            rss_keys = [key for key in csv_row.keys() if key.startswith('WAP')]
            self.rss = np.array([float(csv_row[key]) for key in rss_keys])
        except KeyError as error:
            print("Error while parsing the fingerprint: " + str(error))
            raise error


    def __str__(self) -> str:
        """ Define the str of a fingerprint."""
        return "Fingerprint (%s) : at %s in building %s".format(self.id_, self.position, self.building_id)

    def to_list(self) -> list:
        """ Return a list of the fingerprint attributes."""
        return [self.id_, self.position, self.building_id, self.relative_position,
                self.space_id, self.user_id, self.phone_id, self.rss, self.timestamp]

    def get_rss(self) -> np.ndarray:
        """ Get the RSS vector."""
        return self.rss

    def get_id(self) -> int:
        """ Get the ID."""
        return self.id_

    def get_position(self) -> np.ndarray:
        """ Get the position."""
        return self.position

    def get_timestamp(self) -> int:
        """ Get the timestamp."""
        return self.timestamp

    def get_phone_id(self) -> int:
        """ Get the phone ID."""
        return self.phone_id

    def get_relative_position(self) -> int:
        """ Get the relative position."""
        return self.relative_position

    def get_user_id(self) -> int:
        """ Get the user ID."""
        return self.user_id

    def get_space_id(self) -> int:
        """ Get the space ID."""
        return self.space_id

    def get_building_id(self) -> int:
        """ Get the building ID."""
        return self.building_id

    def fork(self) -> "Fingerprint":
        """ Return a copy of the fingerprint."""
        csv_row = {
            "TIMESTAMP": self.timestamp,
            "PHONEID": self.phone_id,
            "USERID": self.user_id,
            "BUILDINGID": self.building_id,
            "FLOOR": self.position[2],
            "LONGITUDE": self.position[0],
            "LATITUDE": self.position[1],
            "RELATIVEPOSITION": self.relative_position,
            "SPACEID": self.space_id,
            **{"WAP" + "0" * (3 - len(str(index))) + str(index): rssi for index, rssi in enumerate(self.rss)}
        }
        return Fingerprint(self.id_, csv_row)

    def update_rss(self, rss: np.ndarray) -> None:
        """ Update the RSS vector."""
        self.rss = rss
