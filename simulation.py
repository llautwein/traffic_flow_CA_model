import numpy as np
import visualiser
import rule
import random
import cellular_automaton as ca

np.set_printoptions(precision=2)
road_length, num_cars, max_timesteps, max_velocity = 20, 3, 50, 1
initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)
# --- Choose the rule ---

# Rule 184
# r = rule.Rule184(road_length)


# TrafficLights (Fixed Cycle / Green Wave)
# light_pos = [5, 15]
# green_duration = [5, 5]
# red_duration = [5, 5]
# start_red = [False, False]
# offset = [0, 2] # Example offset
# r = rule.TrafficLights(road_length, max_velocity, light_pos, green_duration, red_duration, start_red, offset)
# initial_positions = random.sample(range(road_length), num_cars)


# SelfOrganisedTrafficLights
track_distance = 3
threshold = 3
min_green = 3
max_green = 10
light_pos = [10]
r = rule.SelfOrganisedTrafficLights(road_length, max_velocity, light_pos, track_distance,
                                    threshold, min_green, max_green, braking_probability=0.0)

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