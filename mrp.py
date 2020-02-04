from typing import Mapping
from typevars import _state, _state_transition_matrix, _state_reward_map, _trans_reward_map
import numpy as np
from mp import MP


class MRP(MP):
    def __init__(self,
                 trans_matrix: _state_transition_matrix,
                 trans_rewards: _trans_reward_map = None,
                 state_rewards: _state_reward_map = None,
                 discount_factor: float = 1.):

        assert not (state_rewards is None and trans_rewards is None)

        super().__init__(trans_matrix)

        self.r: np.ndarray = np.empty((self.num_states, self.num_states))
        self.R: np.ndarray = np.empty(self.num_states)
        if trans_rewards:
            self.r: np.ndarray = np.zeros((self.num_states, self.num_states))
            for i in range(self.num_states):
                for j in range(self.num_states):
                    self.r[i][j] = trans_rewards.get((self.state_decoding.get(i),
                                                        self.state_decoding.get(j)), 0.)
            self.R = np.multiply(self.P, self.r).sum(axis=1)
        else:
            for i in range(self.num_states):
                self.R[i] = state_rewards.get(self.state_decoding.get(i))

        self.gamma = discount_factor

    def calculate_value_function(self) -> Mapping[_state, float]:
        sink_states = self.get_sink_state_encodings()
        non_sink_states = list(self.S.difference(sink_states))

        values_vector = np.zeros(self.num_states)
        for i in sink_states:
            values_vector[i] = 0. if np.isclose(self.R[i], 0.) else np.inf

        values_vector[non_sink_states] = np.linalg.inv(
            np.eye(len(non_sink_states)) - self.gamma *
            self.P[non_sink_states][:, non_sink_states]).dot(self.R[non_sink_states])

        return dict(zip(self.state_decoding.values(), values_vector))


if __name__ == '__main__':
    student_tm = {
        'c1': {'c2': 0.5, 'fb': 0.5},
        'c2': {'c3': 0.8, 'sleep': 0.2},
        'c3': {'pass': 0.6, 'pub': 0.4},
        'pass': {'sleep': 1.0},
        'pub': {'c1': 0.2, 'c2': 0.4, 'c3': 0.4},
        'fb': {'c1': 0.1, 'fb': 0.9},
        'sleep': {'sleep': 1.0}
    }

    student_rewards = {
        'c1': -2.,
        'c2': -2.,
        'c3': -2.,
        'pass': 10.,
        'pub': 1.,
        'fb': -1.,
        'sleep': 0.
    }

    student_mrp = MRP(trans_matrix=student_tm,
                      state_rewards=student_rewards,
                      discount_factor=1.)

    values = student_mrp.calculate_value_function()
    print(values)
