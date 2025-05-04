import rule
import cellular_automaton as ca
import numpy as np
import random
import visualiser


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

    def density_vel_flow(self, rule):
        """
        Calculates flow, mean velocity and variance vs. density for given rule
        :param rule: the given rule
        :return: Dictionary containing the results
        """
        num_cars_list = np.arange(1, self.road_length, 1)
        results = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
        for num_cars in num_cars_list:
            metrics = self._run_single_simulation(num_cars, rule)

            results['densities'].append(num_cars/road_length)
            results['flows'].append(metrics["flow"])
            results['velocities'].append(metrics["velocity"])
            results['variances'].append(metrics["variance"])
        return results

    def flow_braking_prob_plot(self, p_values, num_cars):
        results = {'densities': [], 'flows': [], 'velocities': [], 'variances': []}
        for i in range(len(p_values)):
            r = rule.Rule184_random(self.road_length, p_values[i])
            metrics = self._run_single_simulation(num_cars, r)

            results['densities'].append(num_cars / road_length)
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
                sync_rule = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                               [T_phase] * num_lights, [T_phase] * num_lights,
                                               start_red=[False] * num_lights, offset=[0] * num_lights,
                                               braking_probability=braking_probability)

                metrics = self._run_single_simulation(num_cars, sync_rule)
                results['densities'].append(num_cars / road_length)
                results['flows'].append(metrics["flow"])
                results['velocities'].append(metrics["velocity"])
                results['variances'].append(metrics["variance"])
            flows.append(results["flows"])
        return flows

    def traffic_light_offset_analysis(self, num_cars_list, max_velocity,
                                  light_positions, green_durations, red_durations,
                                  start_red, offset_range, braking_probability=None):
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
                results['densities'].append(num_cars / road_length)
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
                results['densities'].append(num_cars / road_length)
                results['flows'].append(metrics["flow"])
                results['velocities'].append(metrics["velocity"])
                results['variances'].append(metrics["variance"])

            flows.append(results["flows"])
        return flows

    # ToDo: implement a method that analyses the behaviour of the self organised strategy
    # ToDo: implement a method that compares all the strategies
    # ToDo: expand density flow diagram method to use different v_max, p_values and plot it in the same graph

max_timesteps = int(5000)
max_velocity = 5


# code to obtain cycle length vs. flow plot for different time delays
# to compare synchronised strategy with green wave strategy using the optimal time delay
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = 10
light_positions = [25, 75, 125, 175]
green_durations = [15, 15, 15, 15]
red_durations = [15, 15, 15, 15]
start_red = [False, False, False, False]
cycle_lengths = np.arange(5, 151, 5)
offsets = [0, 10]
flow = analyser.traffic_light_cycle_flow_offset(num_cars, max_velocity,
                                        light_positions, cycle_lengths,
                                        offsets, 0.1)
visualiser.traffic_light_cycle_flow_delay(cycle_lengths, offsets, flow)


"""
# code to obtain flow vs time delay plot
# to verify optimal time delay for the green wave strategy
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
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
                                  start_red, offset_range, braking_probability)
visualiser.traffic_light_delay_flow_plot(num_cars_list, road_length, offset_range, flow)
"""

"""
#Code to obtain the plot cycle length vs. through flow
# to understand the relation of those parameters for the synchronised strategy
road_length = 200
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
num_cars = [10, 40, 100, 140]
cycle_lengths = np.arange(5, 151, 5)
light_positions = [25, 75, 125, 175]
flow = analyser.traffic_light_cycle_analysis(num_cars, max_velocity, light_positions,
                                             cycle_lengths, 0.1)
visualiser.traffic_light_cycle_flow_sync_plot(num_cars, road_length, cycle_lengths, flow)
"""

"""
#Code to obtain the density flow/mean velocity plot
road_length = 100
analyser = Analyser(road_length, max_timesteps, 5)
visualiser = visualiser.Visualiser()
rule = rule.MaxVelocityRandom(road_length, max_velocity, 0.1)
results = analyser.density_vel_flow(rule)
visualiser.density_meanvel_flow_plot(results)
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