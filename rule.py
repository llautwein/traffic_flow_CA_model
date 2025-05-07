import numpy as np

class Rule:
    """
    Implements a rule for the traffic flow model.
    """
    def __init__(self, road_length):
        self.road_length = road_length
        self.light_positions = []

    def apply_rule(self, positions, velocities, time_step):
        pass

    def get_light_states(self, time_step):
        """
        Returns the state of the lights after the rule logic for the given timestep.
        Should be called *after* apply_rule (or internal state update).

        Returns:
            dict: A dictionary mapping light position (int) to its state (bool: True=Green, False=Red).
                  Returns None or empty dict if the rule has no lights or state tracking.
        """
        return {}
        
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
    Includes a maximum velocity rule, in a non-deterministic setup.
    Driver randomly decrease their speed by 1 with a certain probability.
    """

    def __init__(self, road_length, max_velocity, braking_probability=0):
        super().__init__(road_length)
        self.max_velocity = max_velocity
        self.braking_probability = braking_probability

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

    def get_light_states(self, time_step):
        """ Calculates and returns the state of lights for fixed-cycle rules. """
        states = {}
        if self.light_positions is not None:
            for i, light_pos in enumerate(self.light_positions):
                # Call the existing logic used during apply_rule
                states[light_pos] = self.is_light_green(i, time_step)
        return states

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

class SelfOrganisedTrafficLights(MaxVelocity):
    """
    Implements self-organised, queue-based traffic lights.

    Switches red to green when the number of vehicles within distance d exceeds a threshold.
    Ensures a minimum green time, and returns to red either when no vehicles remain or a maximum green time elapses.
    """
    def __init__(self, road_length, max_velocity,
                 light_positions, d, threshold,
                 min_green, max_green, braking_probability=None):
        super().__init__(road_length, max_velocity)
        self.light_positions = light_positions
        self.d = d
        self.threshold = threshold
        self.min_green = min_green
        self.max_green = max_green
        self.braking_probability = braking_probability
        # initial state: all red
        self.is_green = [False] * len(light_positions)
        self.time_since_change = [0] * len(light_positions)
        self.waiting_time_counter = [0] * len(light_positions)

    def update_light_states(self, positions):
        for i, lp in enumerate(self.light_positions):
            switch = False
            distances = (lp - positions) % self.road_length
            count = int(np.sum((distances > 0) & (distances <= self.d)))
            if not self.is_green[i]:
                self.waiting_time_counter[i] += count
                if self.waiting_time_counter[i] >= self.threshold:
                    # switch to green
                    self.is_green[i] = True
                    switch = True
                    self.time_since_change[i] = 0
                    self.waiting_time_counter[i] = 0
            else:
                if self.time_since_change[i] >= self.min_green:
                    if count == 0 or self.time_since_change[i] >= self.max_green:
                        self.is_green[i] = False
                        switch = True
                        self.time_since_change[i] = 0
                        self.waiting_time_counter[i] = 0
            if not switch:
                self.time_since_change[i] += 1

    def get_light_states(self, time_step):
        """ Returns the currently stored state of self-organized lights. """
        states = {}
        for i, light_pos in enumerate(self.light_positions):
            states[light_pos] = self.is_green[i]  # Return the stored boolean state
        return states

    def apply_rule(self, positions, velocities, time_step):
        # update lights based on current queue lengths
        self.update_light_states(positions)

        # sort for updates
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_velocities = velocities[sorted_indices]

        # apply max velocity rule
        gaps = self.compute_gaps(sorted_positions)
        sorted_velocities = np.minimum(sorted_velocities + 1, self.max_velocity)
        sorted_velocities = np.minimum(sorted_velocities, gaps)
        if self.braking_probability is not None:
            brakes = np.random.rand(len(sorted_velocities)) < self.braking_probability
            sorted_velocities[brakes] = np.maximum(sorted_velocities[brakes] - 1, 0)

        # enforce red lights for self-organised
        for i_lp, lp in enumerate(self.light_positions):
            if not self.is_green[i_lp]:
                for j in range(len(sorted_positions)):
                    distance_to_light = (lp - sorted_positions[j]) % self.road_length
                    if 0 < distance_to_light <= sorted_velocities[j]:
                        sorted_velocities[j] = distance_to_light - 1

        # update positions
        sorted_positions = (sorted_positions + sorted_velocities) % self.road_length
        return sorted_positions, sorted_velocities
