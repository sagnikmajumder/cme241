from typing import Mapping, Set, Any
from typevars import _state, _state_distribution, _state_transition_matrix
from math import isclose


def get_state_set(m: Mapping[_state, Any]) -> Set[_state]:
    return set(m.keys())


def get_default_state_encoding(state_set: Set[_state]) -> Mapping[_state, int]:
    return dict({s: i for i, s in enumerate(state_set)})


def is_valid_probability_distribution(sd: _state_distribution) -> bool:
    is_non_negative = all(p >= 0 for p in sd.values())
    is_sum_equal_one = isclose(sum(sd.values()), 1.)

    return is_non_negative and is_sum_equal_one


def is_valid_transition_matrix(tm: _state_transition_matrix) -> bool:
    state_set = get_state_set(tm)
    state_distributions = list(tm.values())

    is_valid_state_maps = all(get_state_set(sd).issubset(state_set) for sd in state_distributions)
    is_valid_distributions = all(is_valid_probability_distribution(sd) for sd in state_distributions)

    return is_valid_state_maps and is_valid_distributions

