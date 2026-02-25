import logging

from aam_simulation.airspace_preset import get_airspace, get_num_routes
from aam_simulation.simulation import Simulation
from aam_simulation import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("aam_simulation")

def main():
    log.info("Building airspace: %s", config.AIRSPACE)
    airspace = get_airspace(config.AIRSPACE)
    num_routes = get_num_routes(config.AIRSPACE)
    num_uavs = {i: config.UAVS_PER_ROUTE for i in range(1, num_routes + 1)}

    log.info("Creating simulation: %d UAVs/route, %d routes", config.UAVS_PER_ROUTE, num_routes)
    sim = Simulation(airspace, num_uavs, min_lat_sep=config.DEFAULT_MIN_LAT_SEP,
                     min_vert_sep=config.DEFAULT_MIN_VERT_SEP, enable_delays=False)

    log.info("Running for %d seconds...", config.RUN_TIME_SEC)
    sim.run(total_time_sec=config.RUN_TIME_SEC)

    log.info("Simulation complete")
    stats = sim.get_statistics()
    print("Simulation Complete")
    print(f"Throughput: {stats['throughput']}")
    print(f"Average TER: {stats['average_TER']:.3f}" if stats["average_TER"] else "No TER data")
    print(f"Total Conflicts Detected: {stats['num_conflicts']}")
    print(f"Total Collisions Detected: {stats['num_collisions']}")

if __name__ == "__main__":
    main()
