import cellular_automaton as ca
import numpy as np
import visualiser

initial_state = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
max_timesteps = 50

CA184 = ca.Rule184(initial_state, max_timesteps)

traffic_evolution = CA184.simulate()
print(traffic_evolution)

visualiser = visualiser.Visualiser(traffic_evolution)

visualiser.matrix_plot()
visualiser.create_gif()







