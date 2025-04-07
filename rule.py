

import numpy as np

class Rule:
    """
    Implements a rule for the traffic flow model.
    """
    def __init__(self, road_length):
        self.road_length = road_length

    def apply_rule(self, positions, velocities, time_step):
        pass

    def BC(self):
        pass
        
class Rule184(Rule):
    def __init__(self, road_length):
        super().__init__(road_length)

    def apply_rule(self, positions, velocities, time_step):
        """
        Updates car positions and velocities based on Rule 184 logic. (Binary Velocity)
        """
        num_cars = len(positions)

        # Sort positions to process them in order
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        for i in range(num_cars):
            current_pos = sorted_positions[i]
            next_pos = (current_pos + 1) % self.road_length

            # Check if next position is occupied
            if next_pos not in sorted_positions:
                sorted_velocities[i] = 1  # Move forward
                sorted_positions[i] = next_pos
            else:
                sorted_velocities[i] = 0  # Stay still

        return sorted_positions, sorted_velocities


class Rule184_random(Rule):
    def __init__(self, road_length, probability):
        super().__init__(road_length)
        self.probability = probability

    def apply_rule(self, positions, velocities, time_step):
        """
        Updates car positions and velocities based on Rule 184. Drivers occasionally stop due to a random event.
        """
        num_cars = len(positions)

        # Sort positions to process them in order
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        for i in range(num_cars):
            if np.random.random() > self.probability:
                current_pos = sorted_positions[i]
                next_pos = (current_pos + 1) % self.road_length  # Wrap around (circular road)

                # Check if next position is occupied
                if next_pos not in sorted_positions:
                    sorted_velocities[i] = 1
                    sorted_positions[i] = next_pos
                else:
                    sorted_velocities[i] = 0
            else:
                sorted_velocities[i] = 0

        return sorted_positions, sorted_velocities

class MaxVelocity(Rule):
    """
    Includes a maximum velocity rule, in a deterministic setup.
    """
    def __init__(self, road_length, max_velocity):
        super().__init__(road_length)
        self.max_velocity = max_velocity

    def compute_gaps(self, current_positions):
        # gap to preceding vehicle
        gap = np.roll(current_positions, -1) - current_positions - np.ones(len(current_positions))
        gap[-1] += self.road_length
        return gap

    def apply_rule(self, positions, velocities, time_step):
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        # Gaps between each car and its preceding vehicle
        gaps = self.compute_gaps(sorted_positions)

        # Increases the velocity by one, except when max velocity is reached
        sorted_velocities = np.minimum(sorted_velocities + 1, self.max_velocity)

        # Ensures that cars don't collide
        sorted_velocities = np.minimum(sorted_velocities, gaps)

        # Updates the positions
        sorted_positions = (sorted_positions + sorted_velocities) % self.road_length
        return sorted_positions, sorted_velocities


class MaxVelocityRandom(Rule):
    """
    Includes a maximum velocity rule, in a non-deterministic setup.
    """

    def __init__(self, road_length, max_velocity, braking_probality):
        super().__init__(road_length)
        self.max_velocity = max_velocity
        self.braking_probability = braking_probality

    def compute_gaps(self, current_positions):
        gap = np.roll(current_positions, -1) - current_positions - np.ones(len(current_positions))
        gap[-1] += self.road_length
        return gap

    def apply_rule(self, positions, velocities, time_step):
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        # Gaps between each car and its preceding vehicle
        gaps = self.compute_gaps(sorted_positions)

        # Increases the velocity by one, except when max velocity is reached
        sorted_velocities = np.minimum(sorted_velocities + 1, self.max_velocity)

        # Includes random braking by drivers
        braking_events = np.random.rand(len(sorted_velocities)) < self.braking_probability
        sorted_velocities[braking_events] = np.maximum(sorted_velocities[braking_events] - 1, 0)

        # Ensures that cars don't collide
        sorted_velocities = np.minimum(sorted_velocities, gaps)

        # Updates the positions
        sorted_positions = (sorted_positions + sorted_velocities) % self.road_length
        return sorted_positions, sorted_velocities

class MaxVelocityTrafficLights(MaxVelocity):
    """
    Implements the max velocity rule with traffic lights
    """
    def __init__(self, road_length, max_velocity, light_position, green_duration, red_duration):
        super().__init__(road_length, max_velocity)
        self.light_position = light_position
        self.green_duration = green_duration
        self.red_duration = red_duration
        self.cycle_length = green_duration + red_duration

    def is_light_green(self, time_step):
        cycle_time = time_step % self.cycle_length
        return cycle_time < self.green_duration

    def apply_rule(self, positions, velocities, time_step):
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        # Gaps between each car and its preceding vehicle
        gaps = self.compute_gaps(sorted_positions)

        # Increases the velocity by one, except when max velocity is reached
        sorted_velocities = np.minimum(sorted_velocities + 1, self.max_velocity)

        # Ensures that cars don't collide
        sorted_velocities = np.minimum(sorted_velocities, gaps)

        if not self.is_light_green(time_step):
            for i in range(len(sorted_positions)):
                distance_to_light = (self.light_position - sorted_positions[i]) % self.road_length
                if 0 < distance_to_light <= sorted_velocities[i]:
                    sorted_velocities[i] = distance_to_light-1

        # Updates the positions
        sorted_positions = (sorted_positions + sorted_velocities) % self.road_length
        return sorted_positions, sorted_velocities



