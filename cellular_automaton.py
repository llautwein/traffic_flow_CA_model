import numpy as np

"""
This class implements the skeleton for a cellular automaton. It is based on a position-velocity
approach, where the positions and velocities of each car are tracked and updated.
"""

class CellularAutomaton:
    def __init__(self, initial_positions, initial_velocities, road_length, max_timesteps):
        self.road_length = road_length
        self.max_timesteps = max_timesteps
        self.positions = initial_positions
        self.velocities = initial_velocities
        self.traffic_evolution = np.zeros((self.max_timesteps, self.road_length))

    def simulate(self, rule):
        for t in range(self.max_timesteps):
            self.update_traffic_evolution(t)
            current_positions = np.copy(self.positions)
            current_velocities = np.copy(self.velocities)

            next_positions, next_velocities = rule.apply_rule(current_positions, current_velocities)
            self.positions = next_positions
            self.velocities = next_velocities

        return self.traffic_evolution

    def update_traffic_evolution(self, t):
        for i in self.positions:
            self.traffic_evolution[t, int(i)] = 1
