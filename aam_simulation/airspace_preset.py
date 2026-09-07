from aam_simulation.airspace import Airspace
from aam_simulation import config

STANDARD = {
    "ref_lat": 28.468,
    "vertiports": [
        {"name": "A", "lat": 28.32, "lon": -82.20},
        {"name": "B", "lat": 28.86, "lon": -81.64},
        {"name": "C", "lat": 28.63, "lon": -81.36},
        {"name": "D", "lat": 28.41, "lon": -80.70},
        {"name": "E", "lat": 28.12, "lon": -81.65},
    ],
    "corridors": [
        {"start": "A", "end": "B", "alt_ab": 3000.0, "alt_ba": 3200.0},
        {"start": "A", "end": "A-C", "alt_ab": 3000.0, "alt_ba": 3200.0},
        {"start": "A-C", "end": "C", "alt_ab": 3000.0, "alt_ba": 3200.0},
        {"start": "B", "end": "C", "alt_ab": 3000.0, "alt_ba": 3200.0},
        {"start": "B", "end": "E", "alt_ab": 3500.0, "alt_ba": 3700.0},
        {"start": "C", "end": "D", "alt_ab": 3000.0, "alt_ba": 3200.0},
        {"start": "C", "end": "E", "alt_ab": 3000.0, "alt_ba": 3200.0},
    ],
    "split_merge_points": [
        {"name": "A-C", "lat": 28.63, "lon": -81.86, "vert_a": "A", "vert_b": "C"},
    ],
    "routes": [
        {"id": 1, "waypoints": ["A", "B"], "alternate": "C"},
        {"id": 2, "waypoints": ["A", "SM:A-C", "C"], "alternate": "B"},
        {"id": 3, "waypoints": ["B", "C"], "alternate": "E"},
        {"id": 4, "waypoints": ["B", "E"], "alternate": "C"},
        {"id": 5, "waypoints": ["C", "D"], "alternate": "E"},
        {"id": 6, "waypoints": ["C", "E"], "alternate": "B"},
    ],
}

SIMPLE = {
    "ref_lat": 28.99,
    "vertiports": [
        {"name": "A", "lat": 28.77, "lon": -81.57},
        {"name": "B", "lat": 29.02, "lon": -81.33},
        {"name": "C", "lat": 29.17, "lon": -81.05},
    ],
    "corridors": [
        {"start": "A", "end": "B", "alt_ab": 3000.0, "alt_ba": 3200.0},
        {"start": "B", "end": "C", "alt_ab": 3000.0, "alt_ba": 3200.0},
    ],
    "split_merge_points": [],
    "routes": [
        {"id": 1, "waypoints": ["A", "B"], "alternate": "C"},
        {"id": 2, "waypoints": ["B", "C"], "alternate": "A"},
    ],
}

PRESETS = {
    "standard": STANDARD,
    "simple": SIMPLE,
}

def build_airspace(definition: dict) -> Airspace:
    airspace = Airspace(definition["ref_lat"])

    for v in definition["vertiports"]:
        airspace.add_vertiport(name=v["name"], lat=v["lat"], lon=v["lon"],
                               num_charge_stations=config.CHARGE_STATIONS)

    for s in definition["split_merge_points"]:
        airspace.add_split_merge_point(name=s["name"], lat=s["lat"], lon=s["lon"],
                                       vert_a_name=s["vert_a"], vert_b_name=s["vert_b"])

    for c in definition["corridors"]:
        airspace.add_corridor(start_name=c["start"], end_name=c["end"], alt_ab=c["alt_ab"], alt_ba=c["alt_ba"])

    for r in definition["routes"]:
        airspace.define_route(route_id=r["id"], waypoint_names=r["waypoints"], alternate_vert_name=r["alternate"])

    return airspace

def get_airspace(name: str) -> Airspace:
    return build_airspace(PRESETS[name.lower()])

def get_num_routes(name: str) -> int:
    return len(PRESETS[name.lower()]["routes"])
