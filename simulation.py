import cellular_automaton as ca
import numpy as np
import visualiser
import rule

initial_state = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
max_timesteps = 50
CA = ca.CellularAutomaton1d(initial_state, max_timesteps)

rule = rule.Rule184()
traffic_evolution = CA.simulate(rule)
print(traffic_evolution)

visualiser = visualiser.Visualiser(traffic_evolution)

visualiser.matrix_plot()
visualiser.create_gif()







