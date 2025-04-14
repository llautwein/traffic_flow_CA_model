import numpy as np
import visualiser
import rule
import random
import cellular_automaton as ca

np.set_printoptions(precision=2)
road_length, num_cars, max_timesteps, max_velocity = 20, 7, 50, 1

light_pos, green_duration, red_duration, start_red = None, None, None, None
light_pos, green_duration, red_duration, start_red = [5, 15], [5, 5], [5, 5], [False, False]
"""
light position is a list of traffic light positions
green and red duration the respective duration of the cycle
start red is a list of booleans indicating if the traffic light starts with red or green
all these lists should be stated if traffic light are used.
"""

initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)

#r = rule.MaxVelocity(road_length, max_velocity)
#r = rule.SynchronisedTrafficLights(road_length, max_velocity, light_pos, green_duration, red_duration, start_red)
offset = [0, 2]
r = rule.TrafficLights(road_length, max_velocity, light_pos, green_duration, red_duration,
                                offset, start_red)
automaton = ca.CellularAutomaton(initial_positions, initial_velocities,
                                 road_length, max_timesteps, 0, road_length-1)

(traffic_evolution, space_mean_velocities, local_variance_velocity,
 local_densities, local_flows) = automaton.simulate(r)
visualiser = visualiser.Visualiser()

visualiser.matrix_plot(traffic_evolution, light_pos, green_duration, red_duration, start_red, offset)
visualiser.create_gif(traffic_evolution, light_pos, green_duration, red_duration, start_red)
print(f"Local (space) mean velocities (per timestep): {space_mean_velocities}")
print(f"Local densities (per timestep): {local_densities}")
print(f"Local flows (per timestep): {local_flows}")

