import numpy as np

class Fingerprint():

    def __init__(self, id: int, csv_row: dict) -> None:
        """ Initialize the fingerprint."""
        try:
            self.id = id
            self.timestamp = int(csv_row['TIMESTAMP'])
            self.user_id = int(csv_row['USERID'])
            self.phone_id = int(csv_row['PHONEID'])
            self.space_id = int(csv_row['SPACEID'])
            self.relative_position = int(csv_row['RELATIVEPOSITION'])
            self.building_id = int(csv_row['BUILDINGID'])
            self.position = np.array([float(csv_row['LONGITUDE']), float(csv_row['LATITUDE']), float(csv_row['FLOOR'])])
            rss_keys = [key for key in csv_row.keys() if key.startswith('WAP')]
            self.rss = np.array([float(csv_row[key]) for key in rss_keys])
        except KeyError:
            print("CSV_row is not well formated.")
            return None
    
    def __str__(self) -> str:
        """ Define the str of a fingerprint."""
        return "Fingerprint (%s) : at %s in building %s".format(self.id, self.position, self.building)
    
    def to_list(self) -> list:
        return [self.id, self.position, self.building_id, self.relative_position, self.space_id, self.user_id, self.phone_id, self.rss, self.timestamp]

    def get_rss(self) -> np.ndarray:
        """ Get the RSS vector."""
        return self.rss
    
    def get_id(self) -> int:
        """ Get the ID."""
        return self.id
    
    def get_position(self) -> np.ndarray:
        """ Get the position."""
        return self.position
    
    def get_timestamp(self) -> int:
        return self.timestamp

    def get_phone_id(self) -> int:
        return self.phone_id
    
    def get_relative_position(self) -> int:
        return self.relative_position
    
    def get_user_id(self) -> int:
        return self.user_id
    
    def get_space_id(self) -> int:
        return self.space_id
    
    def get_building_id(self) -> int:
        return self.building_id
    
    def fork(self) -> "Fingerprint":
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
        return Fingerprint(self.id, csv_row)
    
    def update_rss(self, rss: np.ndarray) -> None:
        """ Update the RSS vector."""
        self.rss = rss
    
        
        
        

