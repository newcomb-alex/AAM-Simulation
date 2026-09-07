from typing import Optional
import numpy as np
from aam_simulation.sim_utils import unit_vector
from aam_simulation import config

class Corridor:
    def __init__(self,
                 start_vertiport, # type : Vertiport
                 end_vertiport, # type : Vertiport
                 altitude_ab: float,
                 altitude_ba: float):
        self.start = start_vertiport
        self.end = end_vertiport
        self.altitude_ab = altitude_ab
        self.altitude_ba = altitude_ba
        
        self.start_pos = self.start.position
        self.end_pos = self.end.position

        self.req_horiz_dist_ab = self._compute_req_horiz_dist(altitude_ab)
        self.req_horiz_dist_ba = self._compute_req_horiz_dist(altitude_ba)

    def _compute_req_horiz_dist(self, cruise_altitude: float) -> float:
        if cruise_altitude > 0:
            t_descent_sec = cruise_altitude / config.DESCENT_RATE_FPS
            avg_speed_ftps = (config.CRUISE_SPEED_KT * config.KNOTS_TO_FT_PER_SEC) / 2.0
            return avg_speed_ftps * t_descent_sec
        return 0.0

    def get_req_horiz_dist_for_descent(self, origin, destination) -> float:
        if origin.name == self.start.name and destination.name == self.end.name:
            return self.req_horiz_dist_ab
        return self.req_horiz_dist_ba

    def get_segment_info(self, origin, destination) -> Optional[tuple]:
        """
        If origin and destination matches this corridor’s endpoints,
        return (altitude_ft, unit_heading_vector (3D np.array)).
        Otherwise return None.
        """

        if origin.name == self.start.name and destination.name == self.end.name:
            altitude = self.altitude_ab
            direction_vec = self.end_pos - self.start_pos
        elif origin.name == self.end.name and destination.name == self.start.name:
            altitude = self.altitude_ba
            direction_vec = self.start_pos - self.end_pos
        else:
            return None
        
        heading_2d = unit_vector(np.array([direction_vec[0], direction_vec[1]]))
        heading_3d = np.array([heading_2d[0], heading_2d[1], 0.0])
        return altitude, heading_3d
    
class SplitMergePoint:
    def __init__(self, name: str, lat: float, lon: float, ref_lat: float, vertiport_a, vertiport_b):
        from aam_simulation.sim_utils import latlon_to_cartesian

        self.name = name
        self.lat = lat
        self.lon = lon
        self.position = latlon_to_cartesian(lat, lon, ref_lat)
        self.vertiport_a = vertiport_a
        self.vertiport_b = vertiport_b
        # altitude is implicitly determined by whichever corridor is used