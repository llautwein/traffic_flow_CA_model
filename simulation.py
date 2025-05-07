import numpy as np
import visualiser
import rule
import random
import cellular_automaton as ca

np.set_printoptions(precision=2)
road_length, num_cars, max_timesteps, max_velocity = 30, 3, 100, 1
initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)
# --- Choose the rule ---

# TrafficLights (Fixed Cycle / Green Wave)
light_pos = [5, 15, 25]
green_duration = [10, 10, 10]
red_duration = [10, 10, 10]
start_red = [False, False, False]
offset = [0, 0, 0]
r = rule.TrafficLights(road_length, max_velocity, light_pos,
                       green_duration, red_duration, start_red, offset, 0.1)


"""
# SelfOrganisedTrafficLights
track_distance = 3
threshold = 5
min_green = 10
max_green = 15
light_pos = [5, 15, 25]
r = rule.SelfOrganisedTrafficLights(road_length, max_velocity, light_pos, track_distance,
                                    threshold, min_green, max_green, braking_probability=0.1)
"""

# --- Setup Automaton and Simulate ---
automaton = ca.CellularAutomaton(np.array(initial_positions), np.array(initial_velocities),
                                 road_length, max_timesteps, 0, road_length-1)

(traffic_evolution, space_mean_velocities, local_variance_velocity,
 local_densities, local_flows, light_state_history) = automaton.simulate(r) # Get history

vis = visualiser.Visualiser()

vis.create_gif(traffic_evolution, r.light_positions, light_state_history)
vis.matrix_plot(traffic_evolution, r.light_positions, light_state_history)

print(f"Local (space) mean velocities (per timestep): {space_mean_velocities}")
print(f"Local flows (per timestep): {local_flows}")