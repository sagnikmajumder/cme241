from typing import Generic, Set, Mapping
from typevars import _state, _state_distribution, _state_transition_matrix
from typeutils import get_state_set, is_valid_transition_matrix, get_default_state_encoding
import numpy as np


class MP(Generic[_state]):
    def __init__(self,
                 tm: _state_transition_matrix) -> None:

        assert(is_valid_transition_matrix(tm))

        self.state_space: Set[_state] = get_state_set(tm)
        self.state_encoding: Mapping[_state, int] = get_default_state_encoding(self.state_space)
        self.state_decoding: Mapping[int, _state] = dict(enumerate(self.state_space))
        self.num_states: int = len(self.state_space)

        self.S: np.ndarray = np.arange(self.num_states)
        self.P: np.ndarray = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.P[i][j] = tm.get(
                    self.state_decoding.get(i)).get(
                    self.state_decoding.get(j), 0.)

    def generate_stationary_distribution(self) -> _state_distribution:
        eigenvals, eigenvecs = np.linalg.eig(self.P)
        evec_one = eigenvecs[:, np.where(np.isclose(eigenvals, 1))].flatten()
        normalized_evec = np.abs(evec_one / np.sum(evec_one))
        return dict(zip(self.state_decoding.values(), normalized_evec))

