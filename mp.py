from typing import Generic, Sequence, Set, Mapping
from typevars import _state, _state_distribution, _state_transition_matrix
from typeutils import get_state_set, is_valid_transition_matrix, get_default_state_encoding
import numpy as np


class MP(Generic[_state]):
    def __init__(self,
                 trans_matrix: _state_transition_matrix) -> None:

        assert(is_valid_transition_matrix(trans_matrix))

        self.state_space: Set[_state] = get_state_set(trans_matrix)
        self.num_states: int = len(self.state_space)

        self.state_encoding: Mapping[_state, int] = get_default_state_encoding(self.state_space)
        self.state_decoding: Mapping[int, _state] = dict(enumerate(self.state_space))

        self.S: Set[int] = set(np.arange(self.num_states))
        self.P: np.ndarray = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.P[i][j] = trans_matrix.get(
                    self.state_decoding.get(i)).get(
                    self.state_decoding.get(j), 0.)

    def generate_stationary_distribution(self) -> _state_distribution:
        eigenvals, eigenvecs = np.linalg.eig(self.P)
        evec_one = eigenvecs[:, np.where(np.isclose(eigenvals, 1))].flatten()
        normalized_evec = np.abs(evec_one / np.sum(evec_one))
        return dict(zip(self.state_decoding.values(), normalized_evec))

    def get_sink_state_encodings(self) -> Set[int]:
        return set([i for i in range(self.num_states) if np.isclose(self.P[i][i], 1.)])

    def get_sink_state_decodings(self) -> Sequence[_state]:
        return [self.state_decoding.get(i) for i in self.get_sink_state_encodings()]


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

    student_mp = MP(student_tm)

    stationary = student_mp.generate_stationary_distribution()
    print(stationary)

    sink_states = student_mp.get_sink_state_decodings()
    print(sink_states)
