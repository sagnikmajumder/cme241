from typing import TypeVar, Mapping, Tuple, Callable

_state = TypeVar('_state')
_state_distribution = Mapping[_state, float]
_state_transition_matrix = Mapping[_state, _state_distribution]

_action = TypeVar('_action')
_state_action_tuple = Tuple[_state, _action]
_state_action_transition_matrix = Mapping[_state_action_tuple, _state_distribution]
