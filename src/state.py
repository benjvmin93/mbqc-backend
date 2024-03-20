from enum import Enum
import numpy as np


class State(Enum):
    ZERO = 0
    ONE = 1
    PLUS = 2
    MINUS = 3
    iMINUS = 4
    iPLUS = 5


def _get_state(state: State) -> np.ndarray:
    """
    Build the given state.
    """
    S = np.zeros((2,), dtype=complex)
    match state:
        case State.ZERO:
            S[0] = 1
        case State.ONE:
            S[1] = 1
        case State.PLUS:
            S[0] = S[1] = 1 / np.sqrt(2)
        case State.MINUS:
            S[0] = 1
            S[1] = -1
            S /= np.sqrt(2)
        case State.iMINUS:
            S[0] = 1
            S[1] = -1j
            S /= np.sqrt(2)
        case State.iPLUS:
            S[0] = 1
            S[1] = 1j
            S /= np.sqrt(2)
    return S


def _tensor(left: State, right: State):
    return np.kron(left, right)


zero = _get_state(State.ZERO)
one = _get_state(State.ONE)
plus = _get_state(State.PLUS)
minus = _get_state(State.MINUS)
iplus = _get_state(State.iPLUS)
iminus = _get_state(State.iMINUS)
