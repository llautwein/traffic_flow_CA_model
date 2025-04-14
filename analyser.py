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
                r = rule.TrafficLights(self.road_length, max_velocity, light_position,
                                                  [T], [T])
                (traffic_evolution, space_mean_velocities, variance_velocity,
                 local_densities, local_flows) = automaton.simulate(r)
                flow.append(np.mean(local_flows[1:]))
            flow_lists.append(flow)
        return flow_lists

    def traffic_light_offset_flow(self, num_cars_list, max_velocity,
                                  light_positions, green_durations, red_durations,
                                  start_red, offset_range, braking_probability=None):
        # computes the through flow for different densities in dependence of the traffic lights offset
        flow_lists = []
        for num_cars in num_cars_list:
            flow = []
            for i in range(len(offset_range)):
                initial_positions = random.sample(range(self.road_length), num_cars)
                initial_velocities = np.zeros(num_cars)
                automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                                 self.road_length, self.max_timesteps,
                                                 0, self.road_length - 1)
                offset = [(k*offset_range[i]) for k in range(len(light_positions))]
                r = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                       green_durations, red_durations, start_red, offset, braking_probability)
                (traffic_evolution, space_mean_velocities, variance_velocity,
                 local_densities, local_flows) = automaton.simulate(r)
                flow.append(np.mean(local_flows[1:]))
            flow_lists.append(flow)
        return flow_lists

    def traffic_light_cycle_flow_offset(self, num_cars, max_velocity,
                                        light_positions, cycle_lengths,
                                        start_red, offsets, braking_probability=None):
        flow_lists = []
        for i in range(len(offsets)):
            flow = []
            for T in cycle_lengths:
                durations = [T for k in range(len(light_positions))]
                initial_positions = random.sample(range(self.road_length), num_cars)
                initial_velocities = np.zeros(num_cars)
                automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                                 self.road_length, self.max_timesteps,
                                                 0, self.road_length - 1)
                offset = [(k * offsets[i]) for k in range(len(light_positions))]
                r = rule.TrafficLights(self.road_length, max_velocity, light_positions,
                                       durations, durations, start_red,
                                       offset, braking_probability)
                (traffic_evolution, space_mean_velocities, variance_velocity,
                 local_densities, local_flows) = automaton.simulate(r)
                flow.append(np.mean(local_flows[1:]))
            flow_lists.append(flow)
        return flow_lists

max_timesteps = int(10000)
max_velocity = 5
"""
# code to obtain cycle length vs. flow plot for different time delays
road_length = 300
analyser = Analyser(road_length, max_timesteps)
visualiser = visualiser.Visualiser()
num_cars = 30
light_positions = [50, 100, 150, 200]
green_durations = [15, 15, 15, 15]
red_durations = [15, 15, 15, 15]
start_red = [False, False, False, False]
cycle_lengths = np.arange(5, 151, 5)
offsets = [0, 5, 10, 20]
flow = analyser.traffic_light_cycle_flow_offset(num_cars, max_velocity,
                                        light_positions, cycle_lengths,
                                        start_red, offsets, 0.1)
visualiser.traffic_light_cycle_flow_delay(cycle_lengths, offsets, flow)
"""


# code to obtain a flow vs time delay plot
road_length = 300
analyser = Analyser(road_length, max_timesteps)
visualiser = visualiser.Visualiser()
num_cars_list = [20, 25, 30, 35]
light_positions = [50, 100, 150, 200]
green_durations = [15, 15, 15, 15]
red_durations = [15, 15, 15, 15]
start_red = [False, False, False, False]
offset_range = [k for k in range(30)]
braking_probability = 0.1
flow = analyser.traffic_light_offset_flow(num_cars_list, max_velocity,
                                  light_positions, green_durations, red_durations,
                                  start_red, offset_range, braking_probability)

visualiser.traffic_light_delay_flow_plot(num_cars_list, road_length, offset_range, flow)


"""
#Code to obtain the plot cycle length vs. through flow
# observe, how the positive effect can only be observed for relatively low car densities
# eventually, the effect vanishes for moderate to high densities
road_length = 100
analyser = Analyser(road_length, max_timesteps)
visualiser = visualiser.Visualiser()
num_cars = [5, 20, 50, 70]
cycle_lengths = np.arange(5, 151, 5)
light_positions = [50]
flow = analyser.traffic_light_cycle_analysis(num_cars, max_velocity, light_positions, cycle_lengths)
visualiser.traffic_light_cycle_flow_sync_plot(num_cars, road_length, cycle_lengths, flow)
"""

"""
#Code to obtain the density flow/mean velocity plot
road_length = 100
analyser = Analyser(road_length, max_timesteps)
visualiser = visualiser.Visualiser()
rule = rule.MaxVelocityRandom(road_length, max_velocity, 0.1)
density, mean_velocity, var_velocity, flow = analyser.density_vel_flow(rule)
visualiser.density_meanvel_flow_plot(density, mean_velocity, var_velocity, flow)
"""

