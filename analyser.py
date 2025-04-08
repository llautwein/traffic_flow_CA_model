import rule
import cellular_automaton as ca
import numpy as np
import random
import visualiser


class Analyser:
    """
    Class that simulates the model multiple times to collect data for several plots.
    """
    def __init__(self, road_length, max_timesteps):
        self.road_length = road_length
        self.max_timesteps = max_timesteps

    def density_vel_flow(self, rule):
        num_cars_list = np.arange(1, self.road_length, 1)
        density = []
        mean_velocity = []
        var_velocity = []
        flow = []
        for num_cars in num_cars_list:
            density.append(num_cars/self.road_length)
            initial_positions = random.sample(range(self.road_length), num_cars)
            initial_velocities = np.zeros(num_cars)
            automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                 self.road_length, self.max_timesteps, 0, self.road_length-1)
            (traffic_evolution, space_mean_velocities, variance_velocity,
             local_densities, local_flows) = automaton.simulate(rule)
            mean_velocity.append(np.mean(space_mean_velocities[1:]))
            var_velocity.append(np.mean(variance_velocity[1:]))
            flow.append(np.mean(local_flows[1:]))
        return density, mean_velocity, var_velocity, flow

    def traffic_light_cycle_analysis(self, num_cars_list, max_velocity, light_position, cycle_lengths):
        """
        Computes the through-flows for different densities in dependence of the cycle lengths.
        """
        flow_lists = []
        for num_cars in num_cars_list:
            flow = []
            for T in cycle_lengths:
                initial_positions = random.sample(range(self.road_length), num_cars)
                initial_velocities = np.zeros(num_cars)
                automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                                 self.road_length, self.max_timesteps,
                                                 0, self.road_length-1)
                r = rule.MaxVelocityTrafficLights(self.road_length, max_velocity, light_position,
                                                  [T], [T])
                (traffic_evolution, space_mean_velocities, variance_velocity,
                 local_densities, local_flows) = automaton.simulate(r)
                flow.append(np.mean(local_flows[1:]))
            flow_lists.append(flow)
        return flow_lists


road_length = 100
max_timesteps = int(10000)
max_velocity = 5
analyser = Analyser(road_length, max_timesteps)
visualiser = visualiser.Visualiser()
"""
Code to obtain the plot cycle length vs. through flow
num_cars = [5, 20, 50, 70]
cycle_lengths = np.arange(5, 151, 5)
light_position = [50]
flow = analyser.traffic_light_cycle_analysis(num_cars, max_velocity, light_position, cycle_lengths)
visualiser.traffic_light_cycle_plot(num_cars, road_length, cycle_lengths, flow)
"""

"""
Code to obtain the density flow/mean velocity plot
rule = rule.MaxVelocityRandom(road_length, max_velocity, 0.9)
density, mean_velocity, var_velocity, flow = analyser.density_vel_flow(rule)
visualiser.density_meanvel_flow_plot(density, mean_velocity, var_velocity, flow)
"""

