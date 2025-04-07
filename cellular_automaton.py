import numpy as np

"""
This class implements the skeleton for a cellular automaton. It is based on a position-velocity
approach, where the positions and velocities of each car are tracked and updated.
"""

class CellularAutomaton:
    def __init__(self, initial_positions, initial_velocities, 
                 road_length, max_timesteps, detect_start=None, detect_end=None):
        self.road_length = road_length
        self.max_timesteps = max_timesteps
        self.positions = initial_positions
        self.velocities = initial_velocities
        self.traffic_evolution = np.zeros((self.max_timesteps, self.road_length))
        self.local_space_meanVels = np.zeros(self.max_timesteps)
        self.local_densities = np.zeros(self.max_timesteps)
        self.local_flows = np.zeros(self.max_timesteps)
        self.start = detect_start
        self.end = detect_end

    def simulate(self, rule):
        for t in range(self.max_timesteps):
            self.update_traffic_evolution(t)
            current_positions = np.copy(self.positions)
            current_velocities = np.copy(self.velocities)

            # local detector measurements
            if self.start is not None and self.end is not None:
                local_mean_velocity, local_density, local_flow = self.local_measurement(current_positions, current_velocities)
                self.local_space_meanVels[t] = local_mean_velocity
                self.local_densities[t] = local_density
                self.local_flows[t] = local_flow
                    
            next_positions, next_velocities = rule.apply_rule(current_positions, current_velocities)
            self.positions = next_positions
            self.velocities = next_velocities

        return (self.traffic_evolution, self.local_space_meanVels, self.local_densities, self.local_flows)

    def local_measurement(self, current_positions, current_velocities):
        mask = (current_positions >= self.start) & (current_positions <= self.end)
        num_cars = np.sum(mask)
        region_length = self.end - self.start + 1  # inclusive detector region
        local_density = num_cars / region_length
    
        if num_cars > 0:
            detected_velocities = current_velocities[mask]
            local_mean_velocity = np.mean(detected_velocities)
        else:
            local_mean_velocity = 0
    
        local_flow = local_density * local_mean_velocity
        
        return(local_mean_velocity, local_density, local_flow)

    def update_traffic_evolution(self, t):
        for i in self.positions:
            self.traffic_evolution[t, int(i)] = 1
