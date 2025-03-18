import numpy as np

class Rule:
    """
    Implements a rule for the traffic flow model.
    """
    def apply_rule(self, current_state):
        pass

class Rule184(Rule):

    def apply_rule(self, current_state):
        next_state = np.copy(current_state)
        # Indices of cars, which have a zero to their right are eligible to move one step
        movable_cars = (current_state == 1) & (np.roll(current_state, -1) == 0)

        # move those cars forward
        next_state[movable_cars] = 0
        next_state[np.roll(movable_cars, 1)] = 1
        return next_state