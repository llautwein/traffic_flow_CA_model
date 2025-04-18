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
    Driver randomly decrease their speed by 1 with a certain probability.
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

        # Ensures that cars don't collide
        sorted_velocities = np.minimum(sorted_velocities, gaps)

        # Includes random braking by drivers
        braking_events = np.random.rand(len(sorted_velocities)) < self.braking_probability
        sorted_velocities[braking_events] = np.maximum(sorted_velocities[braking_events] - 1, 0)

        # Updates the positions
        sorted_positions = (sorted_positions + sorted_velocities) % self.road_length
        return sorted_positions, sorted_velocities

class TrafficLights(MaxVelocity):
    """
    Implements the max velocity rule with traffic lights
    """
    def __init__(self, road_length, max_velocity,
                 light_positions, green_durations,
                 red_durations, start_red=None, offset=None, braking_probability=None):
        """
        :param start_red: boolean array which indicates if the initial cycle should start with red
        :param offset: array which states the time delay of a traffic lights cycle
        """
        super().__init__(road_length, max_velocity)
        self.light_positions = light_positions
        self.green_durations = green_durations
        self.red_durations = red_durations
        self.braking_probability = braking_probability
        if start_red is not None:
            self.start_red = start_red
        else:
            self.start_red = [False for k in range(len(light_positions))]
        if offset is not None:
            self.offset=offset
        else:
            self.offset = [0 for k in range(len(light_positions))]

    def is_light_green(self, i, time_step):
        cycle_length = self.green_durations[i] + self.red_durations[i]
        cycle_time = (time_step-self.offset[i]) % cycle_length
        if self.start_red[i]:
            return cycle_time >= self.red_durations[i]
        else:
            return cycle_time < self.green_durations[i]

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

        # Includes random braking by drivers
        if self.braking_probability is not None:
            braking_events = np.random.rand(len(sorted_velocities)) < self.braking_probability
            sorted_velocities[braking_events] = np.maximum(sorted_velocities[braking_events] - 1, 0)

        for i in range(len(self.light_positions)):
            light_position = self.light_positions[i]
            if not self.is_light_green(i, time_step):
                for j in range(len(sorted_positions)):
                    distance_to_light = (light_position - sorted_positions[j]) % self.road_length
                    if 0 < distance_to_light <= sorted_velocities[j]:
                        sorted_velocities[j] = distance_to_light-1

        # Updates the positions
        sorted_positions = (sorted_positions + sorted_velocities) % self.road_length
        return sorted_positions, sorted_velocities

# Todo: class for self organised, queue-based traffic lights
