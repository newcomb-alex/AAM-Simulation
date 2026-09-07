from typing import Dict, List
from aam_simulation.entities.vertiport import Vertiport
from aam_simulation.entities.corridor import Corridor, SplitMergePoint
from aam_simulation.entities.route import Route

class Airspace:
    def __init__(self, ref_lat: float):
        """
        ref_lat: reference latitude (degrees) for lat/lon to Cartesian conversion.
        """
        self.ref_lat = ref_lat
        self.vertiports: Dict[str, Vertiport] = {}
        self.corridors: List[Corridor] = []
        self.split_merge_points: List[SplitMergePoint] = []
        self.routes: Dict[int, Route] = {}

    def add_vertiport(self, name: str, lat: float, lon: float, num_charge_stations: int) -> Vertiport:
        """
        Generates a vertiport object and adds the vertiport object to the list of vertiports in the 
        airspace.
        """
        v = Vertiport(name, lat, lon, self.ref_lat, num_charge_stations)
        self.vertiports[name] = v
        return v
    
    def add_split_merge_point(self, name: str, lat: float, lon: float, vert_a_name: str, 
                              vert_b_name: str) -> SplitMergePoint:
        """
        Generates a Split/Merge point object to be used as a waypoint and within routes.
        Adds the SplitMergePoint object into the list of split/merge points.
        """
        v_a = self.vertiports[vert_a_name]
        v_b = self.vertiports[vert_b_name]
        smp = SplitMergePoint(name, lat, lon, self.ref_lat, v_a, v_b)
        self.split_merge_points.append(smp)
        return smp
    
    def _get_waypoint(self, name: str):
        if name in self.vertiports:
            return self.vertiports[name]
        smp = next((s for s in self.split_merge_points if s.name == name), None)
        if smp is not None:
            return smp
        raise ValueError(f"Waypoint '{name}' not found.")

    def add_corridor(self, start_name: str, end_name: str, alt_ab: float, alt_ba: float) -> Corridor:
        """
        Generates a corridor object from the starting and ending waypoint names (vertiport or
        split/merge point), as well as the altitudes of the routes in the corridor.
        Adds the corridor into the list of corridors.
        """
        v_a = self._get_waypoint(start_name)
        v_b = self._get_waypoint(end_name)
        corridor = Corridor(v_a, v_b, alt_ab, alt_ba)
        self.corridors.append(corridor)
        return corridor
    
    def define_route(self, route_id: int, waypoint_names: List[str], alternate_vert_name: str) -> Route:
        """
        Generates a route object based on the route id, waypoints, and alternative vertiport. Adds the 
        route to the list of routes.
        """
        waypoints = []
        for name in waypoint_names:
            if name.startswith("SM:"):
                smp_key = name.split("SM:")[1]
                # Find matching SplitMergePoint by concatenating names or a stored attribute
                smp_obj = next(
                    (s for s in self.split_merge_points if s.name == smp_key),
                    None
                )
                if smp_obj is None:
                    raise ValueError(f"SplitMergePoint '{smp_key}' not found.")
                waypoints.append(smp_obj)
            else:
                if name not in self.vertiports:
                    raise ValueError(f"Vertiport '{name}' not found.")
                waypoints.append(self.vertiports[name])
        
        alt_vert = self.vertiports[alternate_vert_name]
        route = Route(route_id, waypoints, alt_vert)
        self.routes[route_id] = route
        return route