import numpy as np
import visualiser
import rule
import random
import cellular_automaton as ca

np.set_printoptions(precision=2)
road_length = 20
num_cars = 7
max_timesteps = 5
detect_start = road_length//2
detect_end = detect_start+4

initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)

#rule = Rule(road_length)
#rule = Rule184(road_length)
#rule = Rule184_random(road_length, 0.05)
rule = Rule184_max_velocity(road_length, 3)
#rule = Rule184_max_velocity_random(road_length, 3, 0.1)
automaton = CellularAutomaton(initial_positions, initial_velocities, road_length, max_timesteps, detect_start, detect_end)

traffic_evolution, space_mean_velocities, local_densities, local_flows = automaton.simulate(rule)
visualiser = Visualiser(traffic_evolution, detect_start, detect_end)

visualiser.matrix_plot()
visualiser.create_gif()
print(f"Local (space) mean velocities (per timestep): {space_mean_velocities}")
print(f"Local densities (per timestep): {local_densities}")
print(f"Local flows (per timestep): {local_flows}")




