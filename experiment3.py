import csv
from aam_simulation import config
from aam_simulation.airspace_preset import get_airspace, get_num_routes
from aam_simulation.simulation import Simulation

def run_experiment():
    # Experiment 3: Vary minimum separation
    start_lat_sep = 50    # feet
    start_vert_sep = 25   # feet
    lat_increment = 50    # lateral increment per config
    vert_increment = 25   # vertical increment every 5 configs
    n_configs = 30
    runs_per_config = 10
    runtime_sec = config.RUN_TIME_SEC

    # Hold UAV count and delays at control values
    per_route = config.UAVS_PER_ROUTE
    enable_delays = False

    results = []

    for idx in range(n_configs):
        # compute separations
        min_lat_sep = start_lat_sep + idx * lat_increment
        min_vert_sep = start_vert_sep + (idx // 5) * vert_increment

        ter_list = []
        throughput_list = []
        conflict_list = []
        collision_list = []

        for run in range(runs_per_config):
            airspace = get_airspace(config.AIRSPACE)                                                                                                                                                        
            num_routes = get_num_routes(config.AIRSPACE)                                                                                                                                                    
            num_uavs = {i: per_route for i in range(1, num_routes + 1)}   

            # run simulation
            sim = Simulation(
                airspace,
                num_uavs_per_route=num_uavs,
                min_lat_sep=min_lat_sep,
                min_vert_sep=min_vert_sep,
                enable_delays=enable_delays
            )
            sim.run(total_time_sec=runtime_sec)

            stats = sim.get_statistics()
            ter_list.append(stats['average_TER'])
            throughput_list.append(stats['throughput'])
            conflict_list.append(stats['num_conflicts'])
            collision_list.append(stats['num_collisions'])

        # compute averages
        avg_ter = sum(ter_list) / runs_per_config
        avg_throughput = sum(throughput_list) / runs_per_config
        avg_conflicts = sum(conflict_list) / runs_per_config
        avg_collisions = sum(collision_list) / runs_per_config

        results.append((min_lat_sep, min_vert_sep,
                        avg_ter, avg_throughput,
                        avg_conflicts, avg_collisions))

        print(f"Config {idx+1}/{n_configs}: "
              f"lat={min_lat_sep}ft, vert={min_vert_sep}ft → "
              f"TERavg={avg_ter:.3f}, "
              f"throughput={avg_throughput:.1f}, "
              f"conflicts={avg_conflicts:.1f}, "
              f"collisions={avg_collisions:.1f}")

    # write results to CSV
    with open('experiment3_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'min_lat_sep', 'min_vert_sep',
            'avg_TERavg', 'avg_throughput',
            'avg_conflicts', 'avg_collisions'
        ])
        writer.writerows(results)

    print('Experiment 3 complete. Results saved to experiment3_results.csv')

if __name__ == '__main__':
    run_experiment()
