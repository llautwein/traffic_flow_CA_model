import numpy as np
import visualiser
import rule
import random
import cellular_automaton as ca

road_length = 20
num_cars = 10
max_timesteps = 5

initial_positions = random.sample(range(road_length), num_cars)
initial_velocities = np.zeros(num_cars)

#rule = rule.Rule(road_length)
#rule = rule.Rule184_random(road_length, 0.05)
#rule = rule.Rule184_max_velocity(road_length, 2)
rule = rule.Rule184_max_velocity_random(road_length, 3, 0.1)
automaton = ca.CellularAutomaton(initial_positions, initial_velocities, road_length, max_timesteps)

traffic_evolution = automaton.simulate(rule)
visualiser = visualiser.Visualiser(traffic_evolution)

visualiser.matrix_plot()
visualiser.create_gif()







