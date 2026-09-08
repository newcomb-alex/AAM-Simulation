import math
import numpy as np
from typing import Dict
from aam_simulation.entities.vertiport import Vertiport
from aam_simulation.entities.uav import UAV
from aam_simulation.entities.route import Route
from aam_simulation.airspace import Airspace
from aam_simulation import config

class Simulation:
    def __init__(self,
                 airspace: Airspace,
                 num_uavs_per_route: Dict[int, int],
                 min_lat_sep: float,
                 min_vert_sep: float,
                 enable_delays: bool = False):
        self.airspace = airspace
        self.min_lat_sep = min_lat_sep
        self.min_vert_sep = min_vert_sep
        self.enable_delays = enable_delays

        self.uavs: Dict[int, UAV] = {}
        self.active_uav_ids = set()
        self.next_uav_id = 1

        # Instantiate UAVs on each route, enqueue for takeoff at t = 0
        for route_id, count in num_uavs_per_route.items():
            route = airspace.routes[route_id]
            for _ in range(count):
                uav_route = Route(route.route_id, list(route.waypoints), route.alternate_vertiport)
                uav = UAV(self.next_uav_id,
                          uav_route,
                          airspace.corridors,
                          airspace.ref_lat)
                self.uavs[self.next_uav_id] = uav
                origin_name = route.waypoints[0].name
                airspace.vertiports[origin_name].request_takeoff(self.next_uav_id)
                self.active_uav_ids.add(self.next_uav_id)
                self.next_uav_id += 1

        # Statistics
        self.time = 0
        self.conflict_count = 0
        self.total_conflicts = 0
        self.collision_count = 0
        self.total_collisions = 0
        self.throughput_count = 0
        self.ter_list = []

        # Delay tracking
        self.delay_end_time = None

        # Calibrate ideal trip times per route direction
        self.ideal_times = {}  # (origin_name, dest_name) -> ideal trip seconds
        self._calibrate_routes()

    def _calibrate_routes(self):
        """Fly one UAV per route direction uninterrupted to record ideal trip duration."""
        routes_to_calibrate = []
        for route in self.airspace.routes.values():
            routes_to_calibrate.append(Route(route.route_id, list(route.waypoints), route.alternate_vertiport))
            routes_to_calibrate.append(Route(route.route_id, list(reversed(route.waypoints)), route.alternate_vertiport))

        for cal_route in routes_to_calibrate:
            origin_name = cal_route.waypoints[0].name
            dest_name = cal_route.waypoints[-1].name
            key = (origin_name, dest_name)
            if key in self.ideal_times:
                continue

            cal_uav = UAV(0, cal_route, self.airspace.corridors, self.airspace.ref_lat)
            cal_uav.initiate_takeoff()

            for t in range(7200):
                prev_state = cal_uav.state
                cal_uav.update_state(t)
                if prev_state == UAV.STATE_DESCENT and cal_uav.state in {UAV.STATE_CHARGING, UAV.STATE_HOLD}:
                    self.ideal_times[key] = cal_uav.trip_duration
                    # Reset destination vertiport state so main simulation is unaffected
                    dest_vert = cal_uav.destination_vertiport
                    dest_vert.last_landing_time = -float('inf')
                    for station in dest_vert.charge_stations:
                        if station.current_uav == 0:
                            station.available = True
                            station.current_uav = None
                            station.time_remaining = 0
                    break

    def run(self, total_time_sec: int):
        for t in range(total_time_sec):
            self.time = t
            self._process_takeoffs()
            self._update_uavs()
            for uav_id in list(self.active_uav_ids):
                self.uavs[uav_id].update_eta()
            if self.enable_delays and t > 0 and t % (5*60) == 0: # Delays refresh every 5 minutes
                self._evaluate_and_apply_delays()
            for v in self.airspace.vertiports.values():
                v.tick(current_time=self.time)
    
    def _process_takeoffs(self):
        for v in self.airspace.vertiports.values():
            if v.takeoff_queue:
                uav_id = v.takeoff_queue[0]

                # Condition 1: Another UAV is landing within 60 seconds
                imminent_landing = any(
                    uav.destination_vertiport == v and uav.eta is not None and 0.0 < uav.eta < 1.0
                    for uav in self.uavs.values()
                    if uav.state in {UAV.STATE_CLIMB, UAV.STATE_CRUISE, UAV.STATE_DESCENT}
                )
                # Condition 2: Recent landing 
                recent_landing = (self.time - v.last_landing_time) < config.TIME_BETWEEN_LANDING_TAKEOFF

                # Condition 3: Recent takeoff
                recent_takeoff = (self.time - v.last_takeoff_time) < config.TIME_BETWEEN_LANDING_TAKEOFF

                # Condition 4: Global conflict-based delay
                delay_active = self.delay_end_time is not None and self.time < self.delay_end_time

                if not (imminent_landing or recent_landing or recent_takeoff or delay_active):
                    uav = self.uavs[uav_id]
                    uav.initiate_takeoff()
                    v.takeoff_queue.popleft()
                    v.last_takeoff_time = self.time
    
    def _update_uavs(self):
        airborne_ids = []
        # Track vertiports already being descended to, to serialize final approaches
        descending_to = {
            self.uavs[uid].destination_vertiport.name
            for uid in self.active_uav_ids
            if self.uavs[uid].state == UAV.STATE_DESCENT
        }
        for uav_id in list(self.active_uav_ids):
            uav = self.uavs[uav_id]
            prev_state = uav.state
            uav.update_state(self.time)

            # Prevent simultaneous landings: if this UAV just started descent but another
            # is already descending to the same vertiport, hold it at cruise for this tick
            if prev_state == UAV.STATE_CRUISE and uav.state == UAV.STATE_DESCENT:
                dest_name = uav.destination_vertiport.name
                if dest_name in descending_to:
                    uav.state = UAV.STATE_CRUISE  # retry next tick
                else:
                    descending_to.add(dest_name)

            if prev_state == UAV.STATE_CHARGING and uav.state == UAV.STATE_TAXI:
                # Completed one full trip
                self.throughput_count += 1
                actual_min = uav.trip_duration / 60.0
                origin_name = uav.flight_plan[0].name
                dest_name = uav.destination_vertiport.name
                ideal_sec = self.ideal_times.get((origin_name, dest_name))
                ideal_min = ideal_sec / 60.0 if ideal_sec is not None else None
                ter = actual_min / ideal_min if ideal_min else 1.0
                self.ter_list.append(ter)

                # Re-enqueue this UAV for takeoff at its current vertiport
                uav.route.waypoints.reverse()
                uav.route.legs = [
                    (uav.route.waypoints[i], uav.route.waypoints[i+1])
                    for i in range(len(uav.route.waypoints) - 1)
                ]
                uav.flight_plan = uav.route.waypoints.copy()
                uav.current_leg_index = 0
                uav.destination_vertiport = uav.route.waypoints[-1]
                uav.has_deviated = False
                current_vertiport_name = uav.route.waypoints[0].name
                self.airspace.vertiports[current_vertiport_name].request_takeoff(uav_id)

                # Reset its trip duration counter
                uav.trip_duration = 0

            if uav.state in {UAV.STATE_CLIMB, UAV.STATE_CRUISE, UAV.STATE_DESCENT, UAV.STATE_EVASIVE, UAV.STATE_HOLD}:
                airborne_ids.append(uav_id)
        
        # Pairwise conflict detection
        collided_ids = set()
        for i in range(len(airborne_ids)):
            if airborne_ids[i] in collided_ids:
                continue
            u1 = self.uavs[airborne_ids[i]]
            if u1.position[2] == 0.0:
                continue
            for j in range(i+1, len(airborne_ids)):
                if airborne_ids[j] in collided_ids:
                    continue
                u2 = self.uavs[airborne_ids[j]]
                conflict = u1.check_for_conflicts(u2, self.min_lat_sep, self.min_vert_sep)
                if conflict:
                    horiz_d, vert_d = conflict
                    if horiz_d <= (u1.sphere_radius + u2.sphere_radius) and vert_d <= (u1.sphere_radius + u2.sphere_radius):
                        self.collision_count += 1
                        self.total_collisions += 1
                        if self.enable_delays:
                            self.delay_end_time = self.time + config.MIN_DELAY_TIME_S
                        for u in (u1, u2):
                            collided_ids.add(u.id)
                            self._reset_uav_after_collision(u)
                        break  # u1 has collided, skip remaining pairs for u1

                    else:
                        # Only count and respond when conflict first detected
                        if not u1.in_conflict:
                            self.conflict_count += 1
                            self.total_conflicts += 1
                            u1.initiate_evasive_action(u2, self.time, self.min_vert_sep)
                            continue
                        if (u1.evasion_type in {"LEFT", "RIGHT"}
                                and u1.evasion_start_time is not None
                                and (self.time - u1.evasion_start_time) <= 5):
                            u1.notify_secondary_conflict()
        
        # Release HOLD UAVs whose evasive climb is complete once lateral conflict has cleared
        for uav_id in airborne_ids:
            if uav_id in collided_ids:
                continue
            uav = self.uavs[uav_id]
            if uav.state == UAV.STATE_HOLD and uav.in_conflict and not uav.waiting_for_charge:
                still_lateral_conflict = any(
                    uav.check_lateral_conflict(self.uavs[other_id], self.min_lat_sep)
                    for other_id in airborne_ids
                    if other_id != uav_id and other_id not in collided_ids
                )
                if not still_lateral_conflict:
                    uav.in_conflict = False
                    uav.state = UAV.STATE_CRUISE

        # After conflict handling, allow mid-flight deviations (skip UAVs that just collided)
        for uav_id in airborne_ids:
            if uav_id not in collided_ids:
                self.uavs[uav_id].maybe_deviate()

    def _reset_uav_after_collision(self, u: UAV):
        """Resets a UAV to its origin and re-enqueues it for takeoff after a collision."""
        self.active_uav_ids.discard(u.id)

        origin = u.route.waypoints[0]
        u.position = np.array([origin.position[0], origin.position[1], 0.0])
        u.altitude = 0.0
        u.speed = 0.0
        u.state = UAV.STATE_TAXI

        u.current_leg_index = 0
        u.destination_vertiport = u.route.waypoints[-1]
        u.flight_plan = u.route.waypoints.copy()
        u.has_deviated = False

        u.in_conflict = False
        u.evasion_start_time = None
        u.evasion_type = None
        u.evasion_phase = None
        u.original_speed = None
        u.original_altitude = None
        u.original_heading = None
        u.secondary_conflict_detected = False
        u.evasive_climb_target = None

        self.airspace.vertiports[origin.name].request_takeoff(u.id)
        # Re-add to active set so update_state is called each tick (UAV waits in TAXI queue)
        self.active_uav_ids.add(u.id)

    def _evaluate_and_apply_delays(self):
        rate_per_min = self.conflict_count / 5.0
        if rate_per_min >= 1.0 or self.collision_count > 0:
            self.delay_end_time = self.time + config.MIN_DELAY_TIME_S
        self.conflict_count = 0
        self.collision_count = 0
    
    def get_statistics(self):
        avg_ter = sum(self.ter_list) / len(self.ter_list) if self.ter_list else None
        throughput = self.throughput_count / (config.RUN_TIME_SEC / 60)
        return {
            "num_conflicts": self.total_conflicts,
            "num_collisions": self.total_collisions,
            "throughput": throughput,
            "average_TER": avg_ter
        }