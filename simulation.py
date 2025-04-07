import numpy as np

import visualiser
import rule
import random
import cellular_automaton as ca

np.set_printoptions(precision=2)
road_length, num_cars, max_timesteps, max_velocity = 20, 5, 50, 2

light_pos, green_duration, red_duration = None, None, None
light_pos, green_duration, red_duration = [5, 15], [10, 2], [2, 10]

initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)

rule = rule.MaxVelocityTrafficLights(road_length, max_velocity, light_pos, green_duration, red_duration)
automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                 road_length, max_timesteps, 0, road_length-1)

(traffic_evolution, space_mean_velocities, local_variance_velocity,
 local_densities, local_flows) = automaton.simulate(rule)
visualiser = visualiser.Visualiser()

visualiser.matrix_plot(traffic_evolution, light_pos, green_duration, red_duration)
visualiser.create_gif(traffic_evolution, light_pos, green_duration, red_duration)
print(f"Local (space) mean velocities (per timestep): {space_mean_velocities}")
print(f"Local densities (per timestep): {local_densities}")
print(f"Local flows (per timestep): {local_flows}")




