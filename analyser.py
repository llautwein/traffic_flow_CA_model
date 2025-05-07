import rule
import cellular_automaton as ca
import numpy as np
import random
import visualiser
import csv
import pickle

class Analyser:
    """
    Class that simulates the model multiple times to collect data for several plots.
    """
    def __init__(self, road_length, max_timesteps, num_runs_per_point, ):
        self.road_length = road_length
        self.max_timesteps = max_timesteps
        self.num_runs_per_point = num_runs_per_point

    def _run_single_simulation(self, num_cars, rule_instance):
        """
        Helper method: Runs multiple simulations for a given configuration
        and returns averaged metrics (flow, velocity, variance).
        """
        run_flows = []
        run_vels = []
        run_vars = []

        for run in range(self.num_runs_per_point):
            initial_positions = np.sort(random.sample(range(self.road_length), num_cars))
            initial_velocities = np.zeros(num_cars)

            automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                             self.road_length, self.max_timesteps,
                                             0, self.road_length - 1)

            (traffic_evolution, space_mean_velocities, local_variance_velocity,
             local_densities, local_flows, _) = automaton.simulate(rule_instance)

            steady_flows = local_flows[1:]
            steady_vels = space_mean_velocities[1:]
            steady_vars = local_variance_velocity[1:]

            run_flows.append(np.mean(steady_flows))
            run_vels.append(np.mean(steady_vels))
            # Variance can be zero, avoid NaN division if averaging variance itself
            run_vars.append(np.mean(steady_vars) if len(steady_vars) > 0 else 0)

        # Return averages over the runs
        avg_metrics = {
            "flow": np.mean(run_flows),
            "velocity": np.mean(run_vels),
            "variance": np.mean(run_vars)
        }
        return avg_metrics

    def density_vel_flow(self, max_velocity_list, braking_prob_list):
        """
        Calculates flow, mean velocity and variance vs. density for given rule
        :param rule: the given rule
        :return: Dictionary containing the results
        """
        num_cars_list = np.arange(1, self.road_length+1, 1)
        results = {}
        for v_max in max_velocity_list:
            for prob in braking_prob_list:
                r = rule.MaxVelocity(road_length, v_max, prob)
                print(f"Processing vmax={v_max}, prob={prob}")
                prob_label = f"{prob:.2f}"
                label = f"vmax={v_max}, p={prob_label}"
                results[label] = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
                for num_cars in num_cars_list:
                    print(f"Density {num_cars/self.road_length}")
                    metrics = self._run_single_simulation(num_cars, r)
                    results[label]['densities'].append(num_cars/self.road_length)
                    results[label]['flows'].append(metrics["flow"])
                    results[label]['velocities'].append(metrics["velocity"])
                    results[label]['variances'].append(metrics["variance"])
        self.write_pickle(results, "pickle_results/density_vel_flow.pkl")
        return results

    def flow_braking_prob_plot(self, p_values, num_cars):
        """
        Analyses the influence of the braking probability on the average flow
        """
        results = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
        for i in range(len(p_values)):
            r = rule.Rule184_random(self.road_length, p_values[i])
            metrics = self._run_single_simulation(num_cars, r)

            results['densities'].append(num_cars / self.road_length)
            results['flows'].append(metrics["flow"])
            results['velocities'].append(metrics["velocity"])
            results['variances'].append(metrics["variance"])
        return results


    def traffic_light_cycle_analysis(self, num_cars_list, max_velocity, light_positions,
                                     cycle_lengths, braking_probability):
        """
        Analyses flow vs. cycle length for SYNCHRONISED traffic light strategy using several densities
        """
        num_lights = len(light_positions)
        flows = []
        for num_cars in num_cars_list:
            results = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
            for T_phase in cycle_lengths:
                print(f"Processing cycle length {T_phase} for density {num_cars/self.road_length}")
                sync_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                               [T_phase] * num_lights, [T_phase] * num_lights,
                                               start_red=[False] * num_lights, offset=[0] * num_lights,
                                               braking_probability=braking_probability)

                metrics = self._run_single_simulation(num_cars, sync_rule)
                results['densities'].append(num_cars / self.road_length)
                results['flows'].append(metrics["flow"])
                results['velocities'].append(metrics["velocity"])
                results['variances'].append(metrics["variance"])
            flows.append(results["flows"])
        self.write_pickle(flows, "pickle_results/flows_cycle.pkl")
        return flows

    def analyse_green_red_split(self, num_cars, max_velocity, light_positions,
                                total_cycle_lengths, braking_probability):
        results = {}
        num_lights = len(light_positions)
        for T in total_cycle_lengths:
            print(f"Processing cycle length {T}")
            cycle_results = {'green_durations': [], 'red_durations': [], 'flows': []}
            for green_dur in range(1, T):
                red_dur = T - green_dur
                sync_rule = rule.TrafficLights(
                    self.road_length, max_velocity, light_positions,
                    [green_dur] * num_lights, [red_dur] * num_lights,
                    start_red=[False] * num_lights, offset=[0] * num_lights,
                    braking_probability=braking_probability
                )
                metrics = self._run_single_simulation(num_cars, sync_rule)

                cycle_results['green_durations'].append(green_dur)
                cycle_results['red_durations'].append(red_dur)
                cycle_results['flows'].append(metrics["flow"])

            results[T] = cycle_results
        self.write_pickle(results, "pickle_results/green_red_split.pkl")
        return results

    def traffic_light_offset_analysis(self, num_cars_list, max_velocity,
                                  light_positions, green_durations, red_durations,
                                  offset_range, braking_probability=None):
        """
        Analyses flow vs. offset time used to find optimal offset for green wave strategy
        """
        flows = []
        for num_cars in num_cars_list:
            results = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
            for i in range(len(offset_range)):
                offset = [(k * offset_range[i]) for k in range(len(light_positions))]
                sync_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                               green_durations, red_durations, offset=offset,
                                               start_red=[False] * len(light_positions),
                                               braking_probability=braking_probability)

                metrics = self._run_single_simulation(num_cars, sync_rule)
                results['densities'].append(num_cars / self.road_length)
                results['flows'].append(metrics["flow"])
                results['velocities'].append(metrics["velocity"])
                results['variances'].append(metrics["variance"])

            flows.append(results["flows"])
        return flows

    def traffic_light_cycle_flow_offset(self, num_cars, max_velocity,
                                        light_positions, cycle_lengths,
                                        offsets, braking_probability=None):
        """
        Compares green wave strategy with synchronised strategy
        """
        flows = []
        velocities = []
        num_lights = len(light_positions)
        for i in range(len(offsets)):
            results = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
            for T_phase in cycle_lengths:
                offset = [(k * offsets[i]) for k in range(len(light_positions))]
                sync_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                               [T_phase] * num_lights, [T_phase] * num_lights,
                                               start_red=[False] * num_lights, offset=offset,
                                               braking_probability=braking_probability)

                metrics = self._run_single_simulation(num_cars, sync_rule)
                results['densities'].append(num_cars / self.road_length)
                results['flows'].append(metrics["flow"])
                results['velocities'].append(metrics["velocity"])
                results['variances'].append(metrics["variance"])

            flows.append(results["flows"])
            velocities.append(results["velocities"])
        return flows, velocities

    def compare_sync_gw(self, num_cars_list, max_velocity, light_positions,
                        sync_parameters, optimal_offset):

        num_lights = len(light_positions)
        results = {
            "densities": [],
            'gw_flows': [], 'sync_flows': [],
            'gw_velocities': [], 'sync_velocities': []
        }

        sync_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                     [sync_parameters['green_duration']] * num_lights,
                                     [sync_parameters['red_duration']] * num_lights,
                                     [sync_parameters.get('start_red', False)] * num_lights,
                                     offset=[0] * num_lights,
                                     braking_probability=0.1)

        gw_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                     [sync_parameters['green_duration']] * num_lights,
                                     [sync_parameters['red_duration']] * num_lights,
                                     [sync_parameters.get('start_red', False)] * num_lights,
                                     offset=[k*optimal_offset for k in range(num_lights)],
                                     braking_probability=0.1)

        for num_cars in num_cars_list:
            print(f"Processing density={num_cars/self.road_length}")
            results["densities"].append(num_cars / self.road_length)
            gw_metrics = self._run_single_simulation(num_cars, gw_rule)
            sync_metrics = self._run_single_simulation(num_cars, sync_rule)
            results['gw_flows'].append(gw_metrics["flow"])
            results['gw_velocities'].append(gw_metrics["velocity"])
            results['sync_flows'].append(sync_metrics["flow"])
            results['sync_velocities'].append(sync_metrics["velocity"])

        return results

    def analyse_sotl_parameter_onefixed(self, num_cars, max_velocity, light_positions,
                               parameter_to_vary, parameter_values, fixed_sotl_parameters):
        """
        Analyses the influence of one parameter of the self organised strategy by fixing all the other parameters
        :param parameter_to_vary: The parameter out of {threshold, distance, min_green_max_green} to vary
        :param parameter_values: The parameter values of the varied parameter
        :param fixed_sotl_parameters: Dict containing the remaining fixed parameters
        :return: results dict
        """
        results = {'parameter_values': parameter_values, 'flows': []}
        density = num_cars / self.road_length
        for value in parameter_values:
            current_params = fixed_sotl_parameters.copy()
            current_params[parameter_to_vary] = value

            sotl_rule = rule.SelfOrganisedTrafficLights(
                self.road_length, max_velocity, light_positions,
                current_params['d'], current_params['threshold'],
                current_params['min_green'], current_params['max_green'], 0.1
            )

            metrics = self._run_single_simulation(num_cars, sotl_rule)
            results['flows'].append(metrics["flow"])
        return results

    @staticmethod
    def write_csv_grid(data_grid, filename):
        full_path = f"{filename}.csv"
        print(f"Writing 2D data to {full_path}...")
        try:
            with open(full_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data_grid)
            print("Write successful.")
        except Exception as e:
            print(f"Error writing to {full_path}: {e}")

    @staticmethod
    def write_pickle(results, filename):
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def analyse_sotl_parametergrid(self, num_cars, max_velocity, light_positions,
                                   threshold_values, distance_values, fixed_parameters):

        flows_grid = np.zeros((len(threshold_values), len(distance_values)))
        for i, threshold_val in enumerate(threshold_values):
            for j, distance_val in enumerate(distance_values):
                print(f"  Processing Threshold={threshold_val}, Distance={distance_val}")

                sotl_rule = rule.SelfOrganisedTrafficLights(
                    self.road_length, max_velocity, light_positions,
                    d=distance_val,
                    threshold=threshold_val,
                    min_green=fixed_parameters['min_green'],
                    max_green=fixed_parameters['max_green'],
                    braking_probability=fixed_parameters['braking_probability']
                )

                metrics = self._run_single_simulation(num_cars, sotl_rule)
                flows_grid[i, j] = metrics["flow"]
        results = {
            'thresholds': threshold_values,
            'distances': distance_values,
            'flows_grid': flows_grid
        }
        self.write_csv_grid(flows_grid, "data/flows_grid.csv")
        return results

    def compare_gw_sotl(self, num_cars_list, max_velocity, light_positions,
                        gw_parameters, sotl_parameters):

        num_lights = len(light_positions)
        results = {
            "densities": [],
            'gw_flows': [], 'sotl_flows': [],
            'gw_velocities': [], 'sotl_velocities': []
        }

        gw_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                     [gw_parameters['green_duration']] * num_lights,
                                     [gw_parameters['red_duration']] * num_lights,
                                     [gw_parameters.get('start_red', False)] * num_lights,
                                     offset=gw_parameters.get("offset"),
                                     braking_probability=0.1)

        sotl_rule = rule.SelfOrganisedTrafficLights(
            self.road_length, max_velocity, light_positions,
            sotl_parameters['d'], sotl_parameters['threshold'],
            sotl_parameters['min_green'], sotl_parameters['max_green'],
            braking_probability=0.1)

        for num_cars in num_cars_list:
            print(f"Processing density={num_cars/self.road_length}")
            results["densities"].append(num_cars / self.road_length)
            gw_metrics = self._run_single_simulation(num_cars, gw_rule)
            sotl_metrics = self._run_single_simulation(num_cars, sotl_rule)
            results['gw_flows'].append(gw_metrics["flow"])
            results['gw_velocities'].append(gw_metrics["velocity"])
            results['sotl_flows'].append(sotl_metrics["flow"])
            results['sotl_velocities'].append(sotl_metrics["velocity"])

        return results


max_timesteps = int(2000)
max_velocity = 5

"""
# compares gw and sotl strategies across all densities for fixed parameters (density / flow and velocity - plots)
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars_list = np.arange(1, road_length + 1, 1)
light_positions = [25, 75, 125, 175]
gw_parameters = {"green_duration": 10, "red_duration": 10,
                 "offset": [k*10 for k in range(len(light_positions))]}
sotl_parameters = {"d": 10, "threshold":25,
                   "min_green": 10, "max_green":20}
results = analyser.compare_gw_sotl(num_cars_list, max_velocity, light_positions,
                        gw_parameters, sotl_parameters)
visualiser.compare_gw_sotl(results)
"""

"""
# Analyses the influence of distance and threshold parameter in a 2d grid and produces a flow heatmap
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = 140
light_positions = [25, 75, 125, 175]
fixed_parameters = {"min_green": 10, "max_green": 25, "braking_probability": 0.1}
threshold_values = np.arange(1, 40, 1)
distance_values = np.arange(1, 40, 1)
results = analyser.analyse_sotl_parametergrid(num_cars, max_velocity, light_positions,
                                              threshold_values, distance_values, fixed_parameters)
path_to_grid = "data/flows_grid.csv"
visualiser.sotl_parameter_influence_grid(threshold_values, distance_values, path_to_grid)
"""

"""
# analyses the influence of parameters in the sotl strategy by varying one and fixing the other
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = 10
light_positions = [25, 75, 125, 175]
fixed_parameters= {"threshold":4, "min_green":10, "max_green":25}
parameter_to_vary = "d"
parameter_values = np.arange(5, 40, 1)
results = analyser.analyse_sotl_parameter_onefixed(num_cars, max_velocity, light_positions,
                                parameter_to_vary, parameter_values, fixed_parameters)
visualiser.sotl_parameter_influence_onefixed(parameter_to_vary, parameter_values, results["flows"])
"""

"""
# code to obtain cycle length vs. flow plot for different time delays
# to compare synchronised strategy with green wave strategy using the optimal time delay
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = 100
light_positions = [25, 75, 125, 175]
green_durations = [15, 15, 15, 15]
red_durations = [15, 15, 15, 15]
start_red = [False, False, False, False]
cycle_lengths = np.arange(5, 151, 5)
offsets = [0, 10]
flow, velocities = analyser.traffic_light_cycle_flow_offset(num_cars, max_velocity,
                                        light_positions, cycle_lengths,
                                        offsets, 0.1)
visualiser.traffic_light_cycle_flow_delay(cycle_lengths, offsets, flow, velocities)
"""

# code to analyse the influence of green and red proportion
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = 10
light_positions = [25, 75, 125, 175]
total_cycle_lengths = [30, 70, 100]
proportion_results = analyser.analyse_green_red_split(num_cars, max_velocity, light_positions,
                                total_cycle_lengths, 0.1)
results_path = "pickle_results/green_red_split.pkl"
visualiser.red_green_proportion_plot(results_path)

"""
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars_list = np.arange(1, road_length + 1, 1)
sync_parameters = {"green_duration": 15, "red_duration":15}
light_positions = [25, 75, 125, 175]
cycle_lengths = np.arange(5, 151, 5)
optimal_offset = 10
results = analyser.compare_sync_gw(num_cars_list, max_velocity,
                                        light_positions, sync_parameters,
                                        optimal_offset)
visualiser.compare_sync_gw(results)
"""

"""
# code to obtain flow vs time delay plot
# to verify optimal time delay for the green wave strategy
road_length = 200
analyser = Analyser(road_length, max_timesteps, 10)
visualiser = visualiser.Visualiser()
num_cars_list = [10]
light_positions = [25, 75, 125, 175]
green_durations = [15, 15, 15, 15]
red_durations = [15, 15, 15, 15]
start_red = [False, False, False, False]
offset_range = [k for k in range(30)]
braking_probability = 0.1
flow = analyser.traffic_light_offset_analysis(num_cars_list, max_velocity,
                                  light_positions, green_durations, red_durations,
                                  offset_range, braking_probability)
visualiser.traffic_light_delay_flow_plot(num_cars_list, road_length, offset_range, flow)
"""

"""
#Code to obtain the plot cycle length vs. through flow
# to understand the relation of those parameters for the synchronised strategy
road_length = 200
analyser = Analyser(road_length, max_timesteps, 20)
visualiser = visualiser.Visualiser()
num_cars = [10, 40, 100, 140]
cycle_lengths = np.arange(1, 151, 1)
light_positions = [25, 75, 125, 175]
flow = analyser.traffic_light_cycle_analysis(num_cars, max_velocity, light_positions,
                                             cycle_lengths, 0.1)
path_to_flows = "pickle_results/flows_cycle.pkl"
visualiser.traffic_light_cycle_flow_sync_plot(num_cars, road_length, cycle_lengths, path_to_flows)
"""

"""
#Code to obtain the density flow/mean velocity plot
road_length = 100
analyser = Analyser(road_length, max_timesteps, 3)
visualiser = visualiser.Visualiser()
max_velocity_list = [1, 2, 3, 4, 5]
braking_prob_list = [0.0, 0.1, 0.5, 0.9]
#results = analyser.density_vel_flow(max_velocity_list, braking_prob_list)
visualiser.density_meanvel_flow_plot("pickle_results/density_vel_flow.pkl", "vmax=5")
"""

"""
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = 50
p_values = np.linspace(0, 1, 20)
results = analyser.flow_braking_prob_plot(p_values, num_cars)
visualiser.flow_braking_prob(p_values, results["flows"])
"""