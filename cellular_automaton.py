from copy import deepcopy

import numpy as np

class CellularAutomaton:

    def __init__(self, initial_state, max_timesteps):
        self.initial_state = initial_state
        self.length = len(initial_state)
        self.max_timesteps = max_timesteps + 1
        self.states = np.zeros((self.max_timesteps, self.length))
        self.states[0, :] = initial_state

    def simulate(self):
        pass


class Rule184(CellularAutomaton):

    def __init__(self, initial_state, max_timesteps):
        super().__init__(initial_state, max_timesteps)

    def simulate(self):
        for t in range(self.max_timesteps-1):
            current_state = np.copy(self.states[t, :])
            self.states[t + 1, :] = np.copy(self.states[t, :])
            # Indices of cars, which have a zero to their right are eligible to move one step
            movable_cars = (current_state == 1) & (np.roll(current_state, -1) == 0)

            # move those cars forward
            self.states[t+1, movable_cars] = 0
            self.states[t+1, np.roll(movable_cars, 1)] = 1


        return self.states
