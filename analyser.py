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
            mean_velocity.append(space_mean_velocities[-1])
            var_velocity.append(np.mean(variance_velocity))
            flow.append(local_flows[-1])
        return density, mean_velocity, var_velocity, flow


road_length = 100
max_timesteps = 50
max_velocity = 3
analyser = Analyser(road_length, max_timesteps)
rule = rule.MaxVelocity(road_length, max_velocity)
density, mean_velocity, var_velocity, flow = analyser.density_vel_flow(rule)
visualiser = visualiser.Visualiser()
visualiser.density_meanvel_flow_plot(density, mean_velocity, var_velocity, flow)
