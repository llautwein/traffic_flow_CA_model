from copy import deepcopy

import numpy as np

class CellularAutomaton1d:

    def __init__(self, initial_state, max_timesteps):
        self.initial_state = initial_state
        self.length = len(initial_state)
        self.max_timesteps = max_timesteps + 1
        self.states = np.zeros((self.max_timesteps, self.length))
        self.states[0, :] = initial_state

    def simulate(self, rule):
        for t in range(self.max_timesteps-1):
            current_state = np.copy(self.states[t, :])
            next_state = rule.apply_rule(current_state)
            self.states[t+1, :] = np.copy(next_state)

        return self.states