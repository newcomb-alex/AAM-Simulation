import csv
from pathlib import Path
from aam_simulation import config
from aam_simulation.airspace_preset import get_airspace, get_num_routes
from aam_simulation.simulation import Simulation

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def run_experiment():
    increment = 0
    n_configs = 5
    runs_per_config = 1
    runtime_sec = config.RUN_TIME_SEC
    min_lat_sep = config.DEFAULT_MIN_LAT_SEP
    min_vert_sep = config.DEFAULT_MIN_VERT_SEP

    enable_delays = False

    results = []

    for idx in range(n_configs):
        per_route = config.UAVS_PER_ROUTE + (idx * increment)

        conflict_counts = []
        collision_counts = []
        ter_avg_counts = []
        throughput_counts = []

        for run in range(runs_per_config):
            airspace = get_airspace(config.AIRSPACE)                                                                                                                                                        
            num_routes = get_num_routes(config.AIRSPACE)                                                                                                                                                    
            num_uavs = {i: per_route for i in range(1, num_routes + 1)}  

            # 3. Create and run simulation
            sim = Simulation(airspace,
                             num_uavs_per_route=num_uavs,
                             min_lat_sep=min_lat_sep,
                             min_vert_sep=min_vert_sep,
                             enable_delays=enable_delays)
            sim.run(total_time_sec=runtime_sec)

            stats = sim.get_statistics()
            ter_avg_counts.append(stats["average_TER"])
            throughput_counts.append(stats["throughput"])
            conflict_counts.append(stats["num_conflicts"])
            collision_counts.append(stats["num_collisions"])
            
        # average over runs
        avg_ter_rate = sum(ter_avg_counts) / runs_per_config
        avg_throughput_rate = sum(throughput_counts) / runs_per_config
        avg_conflict_rate = sum(conflict_counts) / runs_per_config
        avg_collision_rate = sum(collision_counts) / runs_per_config

        results.append((per_route, avg_ter_rate, avg_throughput_rate, avg_conflict_rate, avg_collision_rate))

        print(f"Done config {idx+1}/{n_configs}: "
              f"UAVs/route={per_route} → "
              f"ter_rate={avg_ter_rate:.3f}, "
              f"thru_rate={avg_throughput_rate:.3f},"
              f"conf_rate={avg_conflict_rate:.3f}, "
              f"coll_rate={avg_collision_rate:.3f}")

    # write out CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "control_experiment_results.csv"

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["UAVs_per_route", "avg_ter_rate", "avg_throughput_rate", "conflicts", "collisions"])
        writer.writerows(results)

    print(f"Experiment complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
