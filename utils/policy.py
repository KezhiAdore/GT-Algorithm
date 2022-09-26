import numpy as np


class Policy:
    """Base policy

    A policy is something giving an action probability distribution given a state of the game
    """

    def __init__(self, player_id, num_actions):
        """Initial a policy

        Args:
            player_id: An id identify the player
            num_actions: The total amount of action in the game
        """

        self._player_id = player_id
        self._num_actions = num_actions

    def action_probabilities(self, state, legal_action_mask=None):
        """ Return a dictionary {action:probability} for all actions

        Args:
            state: A game state
            legal_action_mask: A list show whether an action legal of all actions
            if None, all actions are legal
        Returns:
            A `dict` of `{action:probability}` for the giving game state
        """
        return NotImplemented()

    def __call__(self, state, legal_action_mask=None):
        """Turns the policy into a callable
        Args:
            state: A game state

        Returns:
            A `dict` of `{action:probability}` for the giving game state
        """
        return self.action_probabilities(state, legal_action_mask)


class TabularPolicy(Policy):
    """Tabular policy

    Tabular policy use a table to store the probability of choosing an action given a state of the game
    """

    def __init__(self, player_id, num_actions):
        super().__init__(player_id, num_actions)

        self.state_lookup = {}
        self.action_probability_array = np.ndarray((0, self._num_actions))
        self.legal_action_list = []
        self.epsilon = 1E-9

    def action_probabilities(self, state, legal_action_mask=None):
        """

        Args:
            state: A game state
            legal_action_mask: A list show whether an action legal of all actions

        Returns:
            A dictionary {action:probability} given the state
            if the state in tabular, return the record action probabilities
            else, return random action probabilities and record them in tabular
        """
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]

        if np.sum(legal_action_mask) == 0:
            return {action: 0 for action in range(self._num_actions)}

        state_key = self._state_key(state)
        if state_key in self.state_lookup:
            state_index = self.state_lookup[state_key]
            action_probs = self.action_probability_array[state_index]
            action_probs *= legal_action_mask
            if np.sum(action_probs) > 0:
                action_probs /= np.sum(action_probs)
            else:
                action_probs = np.array(legal_action_mask / np.sum(legal_action_mask))
        else:
            action_probs = np.array(legal_action_mask / np.sum(legal_action_mask))
            self.set_action_probabilities(state, action_probs, legal_action_mask)

        return {action: action_probs[action] for action in range(self._num_actions)}

    def set_action_probabilities(self, state, action_probs, legal_action_mask=None):
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]
        state_key = self._state_key(state)
        if state_key in self.state_lookup:
            state_index = self.state_lookup[state_key]
            self.legal_action_list[state_index] = legal_action_mask
            self.action_probability_array[state_index] = action_probs
        else:
            state_index = len(self.state_lookup)
            self.state_lookup[state_key] = state_index
            self.legal_action_list.append(legal_action_mask)
            self.action_probability_array = np.append(self.action_probability_array,
                                                      np.array(action_probs).reshape(1, -1), axis=0)

    def _state_key(self, state):
        return repr(state)


class RandomPolicy(Policy):

    def __init__(self, player_id, num_actions):
        super(RandomPolicy, self).__init__(player_id, num_actions)

    def action_probabilities(self, state, legal_action_mask=None):
        """

        Args:
            state:
            legal_action_mask:

        Returns:
            Uniform random policy, contain all legal actions, each with the same probability
        """
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]

        if np.sum(legal_action_mask) == 0:
            return {action: 0 for action in range(self._num_actions)}

        action_probs = np.array(legal_action_mask / np.sum(legal_action_mask))
        return {action: action_probs[action] for action in range(self._num_actions)}


if __name__ == "__main__":
    tabular_policy = TabularPolicy("Tabular Test", 4)
    random_policy = RandomPolicy("Random Test", 4)
    for i in range(1000):
        state = np.random.randint(0, 2, (1, 10))
        legal_action_mask = np.random.randint(0, 2, (4))
        tabular_policy(state, legal_action_mask)
        random_policy(state,legal_action_mask)
        if np.random.random() > 0.5:
            action_probs = np.random.random((1, 4))
            tabular_policy.set_action_probabilities(state, action_probs, legal_action_mask)
    print(tabular_policy.action_probability_array)
