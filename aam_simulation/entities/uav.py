import math
import random
import numpy as np
from aam_simulation.sim_utils import unit_vector, signed_angle_between
from aam_simulation.entities.vertiport import Vertiport
from aam_simulation.entities.route import Route
from aam_simulation import config

class UAV:
    # UAV States
    STATE_TAXI = "TAXI"
    STATE_CLIMB = "CLIMB"
    STATE_CRUISE = "CRUISE"
    STATE_DESCENT = "DESCENT"
    STATE_EVASIVE = "EVASIVE"
    STATE_HOLD = "HOLD"
    STATE_CHARGING = "CHARGING"

    def __init__(self,
                 uav_id: int,
                 route: Route,
                 corridor_list: list,
                 ref_lat: float):
        self.id = uav_id
        self.route = route
        self.corridors = corridor_list
        self.ref_lat = ref_lat

        origin_vert = route.waypoints[0]
        self.position = np.array([origin_vert.position[0], origin_vert.position[1], 0.0])
        self.speed = 0.0 # knots
        self.heading = np.array([0.0, 0.0, 0.0])
        self.sphere_radius = config.UAV_RADIUS_FT

        self.state = UAV.STATE_TAXI
        self.waiting_for_charge = False
        self.current_leg_index = 0
        self.altitude = 0.0
        self.destination_vertiport = route.waypoints[-1]
        self.flight_plan = route.waypoints.copy()
        self.eta = None
        self.trip_duration = 0
        self.time_to_charge = 0
        self.has_deviated = False

        # Evasive/conflict tracking
        self.in_conflict = False
        self.evasion_start_time = None
        self.evasion_type = None
        self.original_speed = None
        self.original_altitude = None
        self.original_heading = None
        self.evasion_phase = None
        self.secondary_conflict_detected = False
        self.min_vert_sep = None
        self.evasive_climb_target = None

    def _find_current_corridor(self):
        """Return (corridor, origin, dest, altitude, heading_unit) for the current leg, or None."""
        if self.current_leg_index >= len(self.route.legs):
            return None
        origin, dest = self.route.legs[self.current_leg_index]
        for corridor in self.corridors:
            seg = corridor.get_segment_info(origin, dest)
            if seg:
                altitude, heading_unit = seg
                return corridor, origin, dest, altitude, heading_unit
        return None
    
    def update_state(self, time_step: int):
        """
        Called once per second by Simulation. Updates position, state transitions,
        conflict recovery, charging countdown, and other UAV state attributes.
        """

        # Update trip duration in all active flight phases (reset on takeoff, so taxi wait excluded)
        if self.state in {
            UAV.STATE_CLIMB,
            UAV.STATE_CRUISE,
            UAV.STATE_DESCENT,
            UAV.STATE_EVASIVE,
            UAV.STATE_HOLD,
            UAV.STATE_TAXI
        }:
            self.trip_duration += 1
        
        # 1. CHARGING State
        if self.state == UAV.STATE_CHARGING:
            self.time_to_charge -= 1
            if self.time_to_charge <= 0:
                self.state = UAV.STATE_TAXI
                self.time_to_charge = 0
            return
        
        # 2. TAXI State
        if self.state == UAV.STATE_TAXI:
            self.flight_plan[0].request_takeoff(self.id)
            return # waiting for takeoff clearance
        
        # 3. EVASIVE or HOLD
        if self.state == UAV.STATE_EVASIVE:
            self._handle_evasive(time_step)
            return
        
        if self.state == UAV.STATE_HOLD: 
            # Remain in HOLD until Simulation clears conflict
            if self.waiting_for_charge:
                if self.destination_vertiport.assign_charge_station(self.id):
                    self.state = UAV.STATE_CHARGING
                    self.time_to_charge = config.CHARGE_TIME_SEC
                    self.waiting_for_charge = False
            return
        
        # 4. CLIMB (Diagonal climb + accelerate)
        if self.state == UAV.STATE_CLIMB:
            # First, fetch the corridor we are climbing into:
            corridor_info = self._find_current_corridor()
            if corridor_info is None:
                # No more legs, we must already be at destination (should not happen here)
                self.state = UAV.STATE_DESCENT
                return
            
            _, origin, dest, cruise_altitude, heading_unit = corridor_info

            # a) Vertical climb: 900 ft/min = 15 ft/s
            climb_rate_fps = config.CLIMB_RATE_FPS
            self.altitude += climb_rate_fps
            if self.altitude > cruise_altitude:
                self.altitude = cruise_altitude
            
            # b) Lateral acceleration: linearly from 0 to 115 knots over the same vertical distance
            # Time to climb from 0 to cruise altitude at 15 ft/s: t_climb_sec = cruise_altitude / 15
            if cruise_altitude > 0:
                t_climb_sec = cruise_altitude / climb_rate_fps
                accel_rate_kt_per_sec = config.CRUISE_SPEED_KT / t_climb_sec
            else: 
                accel_rate_kt_per_sec = 0.0

            self.speed += accel_rate_kt_per_sec
            if self.speed > config.CRUISE_SPEED_KT:
                self.speed = config.CRUISE_SPEED_KT
            
            # c) Horizontal movement: move forward by current speed (in ft/s)
            speed_fps = self.speed * config.KNOTS_TO_FT_PER_SEC # convert knots to ft/s
            movement = np.array([heading_unit[0], heading_unit[1], 0.0]) * speed_fps
            self.position += movement

            # d) Once both altitude and speed have reached cruise targets, transition to CRUISE
            if self.altitude >= cruise_altitude and self.speed >= config.CRUISE_SPEED_KT:
                self.altitude = cruise_altitude
                self.speed = config.CRUISE_SPEED_KT
                self.state = UAV.STATE_CRUISE

            return
        
        # 5. CRUISE (constant altitude & speed, check for descent trigger)
        if self.state == UAV.STATE_CRUISE:
            corridor_info = self._find_current_corridor()

            # If we have flown all legs, begin descent immediately
            if corridor_info is None:
                self.state = UAV.STATE_DESCENT
                return
            
            corridor_obj, origin, dest, cruise_altitude, heading_unit = corridor_info

            # Compute how far (2D) we are currently from the next waypoint (dest)
            dest_xy = np.array([dest.position[0], dest.position[1]])
            cur_xy = np.array([self.position[0], self.position[1]])
            horiz_dist_to_dest_ft = np.linalg.norm(dest_xy - cur_xy)

            # Look up the precomputed required horizontal distance to descend from cruise to 0
            required_horiz_dist_for_descent = corridor_obj.get_req_horiz_dist_for_descent(origin, dest)
            
            # If we are within that descent distance and dest is a vertiport, begin diagonal descent
            if isinstance(dest, Vertiport) and horiz_dist_to_dest_ft <= required_horiz_dist_for_descent:
                self.state = UAV.STATE_DESCENT
                return
            
            # Otherwise, remain at cruise altitude & speed, moving forward by the cruise speed each sec
            speed_fps = config.CRUISE_SPEED_KT * config.KNOTS_TO_FT_PER_SEC
            movement = np.array([heading_unit[0], heading_unit[1], 0.0]) * speed_fps
            self.position += movement
            self.altitude = cruise_altitude
            self.speed = config.CRUISE_SPEED_KT

            # Check if arrived at or passed destination (for split/merge point)
            if horiz_dist_to_dest_ft <= speed_fps:
                # Snap to exactly destination
                self.position[0], self.position[1] = dest_xy[0], dest_xy[1]
                if not isinstance(dest, Vertiport): # passed a split/merge point
                    self.current_leg_index += 1
                # Else: we are at a vertiport boundary. If this wasn't caught just before, next tick handles it.
            return 
        
        # 6. Descent (diagonal descent + decelerate)
        if self.state == UAV.STATE_DESCENT:
            corridor_info = self._find_current_corridor()

            # If we have no more legs, that means we are already effectively over the final Vertiport
            # Just keep descending/zeroing out if needed
            if corridor_info is None:
                # We assume we are directly over the destination
                descent_rate_fps = config.DESCENT_RATE_FPS
                if self.altitude > 0.0:
                    self.altitude -= descent_rate_fps
                    if self.altitude < 0.0:
                        self.altitude = 0.0
                        self.speed = 0.0

                if self.altitude <= 0.0:
                    self.altitude = 0.0
                    self.speed = 0.0
                    self._land_at_vertiport(dest_vert = self.destination_vertiport, time_now=time_step)
                return

            # If corridor_info is not None, we still have a "final leg" to travel before the last waypoint
            _, origin, dest, cruise_altitude, heading_unit = corridor_info

            # a) Vertical descent: 7.5 ft/s
            descent_rate_fps = config.DESCENT_RATE_FPS
            self.altitude -= descent_rate_fps
            if self.altitude < 0.0:
                self.altitude = 0.0
            
            # b) Horizontal deceleration: linearly from 115 kt to 0 over the same vertical interval
            if cruise_altitude > 0:
                t_descent_sec = cruise_altitude / descent_rate_fps
                decel_rate_kt_per_sec = config.CRUISE_SPEED_KT / t_descent_sec
            else:
                decel_rate_kt_per_sec = 0.0
            
            self.speed -= decel_rate_kt_per_sec
            if self.speed < 0.0:
                self.speed = 0.0
            
            # c) Horizontal movement: move forward by current speed (ft/s)
            speed_fps = self.speed * config.KNOTS_TO_FT_PER_SEC
            movement = np.array([heading_unit[0], heading_unit[1], 0.0]) * speed_fps
            self.position += movement

            # d) If we have reached or passed the destination 2D point, snap and land
            dest_xy = np.array([dest.position[0], dest.position[1]])
            cur_xy = np.array([self.position[0], self.position[1]])
            horiz_dist_to_dest_ft = np.linalg.norm(dest_xy - cur_xy)

            if horiz_dist_to_dest_ft <= speed_fps or self.altitude <= 0.0:
                # Snap exactly onto the vertiport
                self.position[0], self.position[1] = dest_xy[0], dest_xy[1]
                self.altitude = 0.0
                self.speed = 0.0
                self._land_at_vertiport(dest_vert = dest, time_now=time_step)
            return
        
    def initiate_takeoff(self):
        """
        Called by Simulation when Vertiport grants takeoff clearance. 
        Switch to CLIMB. Set initial heading & reset timers.
        """
        self.state = UAV.STATE_CLIMB
        self.speed = 0.0
        self.altitude = 0.0
        corridor_info = self._find_current_corridor()
        if corridor_info:
            _, _, _, _, heading_unit = corridor_info
            self.heading = np.array([heading_unit[0], heading_unit[1], 0.0])
        self.trip_duration = 0

        # Clear any stale conflict/evasion state from the previous trip
        self.in_conflict = False
        self.evasion_start_time = None
        self.evasion_type = None
        self.evasion_phase = None
        self.original_speed = None
        self.original_altitude = None
        self.original_heading = None
        self.secondary_conflict_detected = False
        self.evasive_climb_target = None

    def check_lateral_conflict(self, other, min_lat_sep: float) -> bool:
        """Returns True if horizontal separation is below the minimum (ignores altitude)."""
        horiz = math.hypot(self.position[0] - other.position[0],
                           self.position[1] - other.position[1])
        return horiz < (min_lat_sep + self.sphere_radius + other.sphere_radius)

    def check_for_conflicts(self, other, min_lat_sep: float, min_vert_sep: float):
        """Returns (horiz_d, vert_d) if in conflict, else None."""
        horiz = math.hypot(self.position[0] - other.position[0],
                           self.position[1] - other.position[1])
        vert = abs(self.altitude - other.altitude)
        if horiz < (min_lat_sep + self.sphere_radius + other.sphere_radius) and vert < min_vert_sep:
            return horiz, vert
        return None
    
    def initiate_evasive_action(self, intruder, current_time: int, min_vert_sep: float):
        """
        Determines the relative location of the intruding UAV and determine the corresponding
        evasive maneuver.
        """
        rel_vec = np.array([
            intruder.position[0] - self.position[0],
            intruder.position[1] - self.position[1],
            intruder.altitude - self.altitude
        ])
        rel_vec_2d = rel_vec[:2]
        heading_2d = self.heading[:2]
        theta = signed_angle_between(heading_2d, rel_vec_2d)
        deg = math.degrees(theta)

        if abs(deg) <= 45:
            designation = "AHEAD"
        elif abs(deg) >= 135:
            designation = "BEHIND"
        elif 45 < deg < 135:
            designation = "LEFT"
        else:
            designation = "RIGHT"
        
        self.original_speed = self.speed
        self.original_altitude = self.altitude
        self.original_heading = self.heading.copy()
        self.evasion_start_time = current_time
        self.in_conflict = True
        self.evasion_type = designation
        self.secondary_conflict_detected = False
        self.min_vert_sep = min_vert_sep

        if designation == "AHEAD":
            if self.speed >= config.MIN_SPEED_TO_SLOW:
                self.speed -= 2.0
                self.state = UAV.STATE_EVASIVE
                self.evasion_phase = "SPEED_REDUCTION"
            else:
                self.evasive_climb_target = self.altitude + min_vert_sep
                self.state = UAV.STATE_EVASIVE
                self.evasion_phase = "EVASIVE_CLIMB"

        elif designation == "BEHIND":
            if self.speed <= config.MAX_SPEED_TO_INCREASE:
                self.speed += 2.0
                self.state = UAV.STATE_EVASIVE
                self.evasion_phase = "SPEED_INCREASE"
            else:
                self.evasive_climb_target = self.altitude + min_vert_sep
                self.state = UAV.STATE_EVASIVE
                self.evasion_phase = "EVASIVE_CLIMB"
        
        elif designation == "LEFT":
            self.state = UAV.STATE_EVASIVE
            self.evasion_phase = "TURN_RIGHT_OUTBOUND"
        
        elif designation == "RIGHT":
            self.state = UAV.STATE_EVASIVE
            self.evasion_phase = "TURN_LEFT_OUTBOUND"
    
    def notify_secondary_conflict(self):
        self.secondary_conflict_detected = True

    def _handle_evasive(self, current_time: int):
        """
        Contains logic to handle the evasive manuever depending on if the intruding UAV is
        Ahead, Behind, Left, or Right. 
        """
        elapsed = current_time - self.evasion_start_time

        if self.evasion_type in {"AHEAD", "BEHIND"}:
            if elapsed >= 60 and self.evasion_phase in {"SPEED_REDUCTION", "SPEED_INCREASE"}:
                self.speed = self.original_speed
                self.evasion_phase = None
                self.in_conflict = False
                self.state = UAV.STATE_CRUISE
            return
        
        corridor_info = self._find_current_corridor()
        if corridor_info:
            _, _, _, _, desired_heading_unit = corridor_info
            desired_heading_2d = np.array([desired_heading_unit[0], desired_heading_unit[1]])
        else:
            desired_heading_2d = np.array([0.0, 0.0])
        current_heading_2d = unit_vector(self.heading[:2])
        turn_rate_out = math.radians(config.TURN_RATE_OUTBOUND)
        turn_rate_rec = math.radians(config.TURN_RATE_RECOVER)

        # Secondary conflict during turn-out → abort and climb
        if self.secondary_conflict_detected and self.evasion_phase in {"TURN_RIGHT_OUTBOUND", "TURN_LEFT_OUTBOUND"}:
            self.secondary_conflict_detected = False
            self.evasion_phase = "TURN_BACK_FOR_CLIMB"

        if self.evasion_phase == "TURN_BACK_FOR_CLIMB":
            orig_h_2d = unit_vector(self.original_heading[:2])
            angle_diff = signed_angle_between(current_heading_2d, orig_h_2d)
            if abs(angle_diff) <= turn_rate_out:
                self.heading = np.array([orig_h_2d[0], orig_h_2d[1], 0.0])
                self.evasive_climb_target = self.altitude + self.min_vert_sep
                self.evasion_phase = "EVASIVE_CLIMB"
            else:
                sign = +1 if angle_diff > 0 else -1
                angle = sign * turn_rate_out
                rot = np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle),  math.cos(angle)]])
                new_h = rot.dot(current_heading_2d)
                self.heading = np.array([new_h[0], new_h[1], 0.0])
            return

        if self.evasion_phase == "EVASIVE_CLIMB":
            self.altitude = min(self.altitude + config.EVASIVE_CLIMB_RATE_FPS, self.evasive_climb_target)
            if self.altitude >= self.evasive_climb_target:
                self.evasion_phase = None
                self.state = UAV.STATE_HOLD
            return

        if elapsed <= 5:
            sign = -1 if self.evasion_type == "LEFT" else +1
            angle = sign * turn_rate_out
            rot = np.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
            new_h = rot.dot(current_heading_2d)
            self.heading = np.array([new_h[0], new_h[1], 0.0])
            return
        
        if elapsed <= 30:
            return
        
        angle_diff = signed_angle_between(current_heading_2d, desired_heading_2d)
        if abs(angle_diff) <= turn_rate_rec:
            self.heading = np.array([desired_heading_2d[0], desired_heading_2d[1], 0.0])
            self.in_conflict = False
            self.evasion_phase = None
            self.state = UAV.STATE_CRUISE
        else:
            sign = +1 if angle_diff > 0 else -1
            angle = sign * turn_rate_rec
            rot = np.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
            new_h = rot.dot(current_heading_2d)
            self.heading = np.array([new_h[0], new_h[1], 0.0])

    def maybe_deviate(self):
        """Randomly deviate a UAV based on probability in config.py"""
        if self.has_deviated:
            return
        if random.random() < config.PROBABILITY_OF_DIVERTION:
            self.has_deviated = True
            if self.current_leg_index < len(self.route.waypoints):
                current_wp = self.route.waypoints[self.current_leg_index]
            else:
                return
            self.flight_plan = [current_wp, self.route.alternate_vertiport]
            self.route.legs = [
                (self.flight_plan[i], self.flight_plan[i+1])
                for i in range(len(self.flight_plan)-1)
            ]
            self.current_leg_index = 0
            self.destination_vertiport = self.route.alternate_vertiport
    
    def update_eta(self):
        """Updates the current ETA of the UAV in minutes"""
        if self.speed is None or self.speed <= 0:
            self.eta = None
            return
        dest_pos = self.destination_vertiport.position
        cur_xy = np.array([self.position[0], self.position[1]])
        horiz_ft = np.linalg.norm(dest_pos - cur_xy)
        nm = horiz_ft / config.FT_IN_NM
        time_hr = nm / self.speed
        self.eta = time_hr * 60.0 # in minutes

    def _land_at_vertiport(self, dest_vert: Vertiport, time_now: int):
        """
        Called when diagonal descent finishes (altitude=0, speed=0, position snapped).
        Switch to CHARGING; Simulation will enqueue at vertiport.
        """
        dest_vert.last_landing_time = time_now
        # try to take an available charge station
        if dest_vert.assign_charge_station(self.id):
            self.state = UAV.STATE_CHARGING
            self.time_to_charge = config.CHARGE_TIME_SEC
            self.waiting_for_charge = False
        else:
            # no charger, wait for one to open
            self.state = UAV.STATE_HOLD
            self.waiting_for_charge = True