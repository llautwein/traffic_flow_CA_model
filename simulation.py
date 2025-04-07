import numpy as np

import visualiser
import rule
import random
import cellular_automaton as ca

np.set_printoptions(precision=2)
road_length = 20
num_cars = 5
max_timesteps = 50

initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)

#rule = Rule(road_length)
#rule = Rule184(road_length)
#rule = Rule184_random(road_length, 0.05)
#rule = rule.MaxVelocity(road_length, 1)
#rule = rule.MaxVelocityRandom(road_length, 3, 0.1)
rule = rule.MaxVelocityTrafficLights(road_length, 2, 10, 5, 10)
automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                 road_length, max_timesteps)

(traffic_evolution, space_mean_velocities, local_variance_velocity,
 local_densities, local_flows) = automaton.simulate(rule)
visualiser = visualiser.Visualiser()

visualiser.matrix_plot(traffic_evolution)
visualiser.create_gif(traffic_evolution)
print(f"Local (space) mean velocities (per timestep): {space_mean_velocities}")
print(f"Local densities (per timestep): {local_densities}")
print(f"Local flows (per timestep): {local_flows}")




