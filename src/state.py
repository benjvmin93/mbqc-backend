from enum import Enum
import numpy as np


class State(Enum):
    ZERO = 0
    ONE = 1
    PLUS = 2
    MINUS = 3
    iMINUS = 4
    iPLUS = 5


def _build_state(state: State, Nqubits: int = 1) -> np.ndarray:
    """
    Build the state according to the number of qubits.
    """
    S = np.zeros((2,) * Nqubits, dtype=complex)
    shape_0 = (0,) * Nqubits
    shape_1 = (1,) * Nqubits
    match state:
        case State.ZERO:
            S[shape_0] = 1
        case State.ONE:
            S[shape_1] = 1
        case State.PLUS:
            S[shape_0] = S[shape_1] = 1 / np.sqrt(2)
        case State.MINUS:
            S[shape_0] = 1
            S[shape_1] = -1
            S /= np.sqrt(2)
        case State.iMINUS:
            S[shape_0] = 1
            S[shape_1] = -1j
            S /= np.sqrt(2)
        case State.iPLUS:
            S[shape_0] = 1
            S[shape_1] = 1j
            S /= np.sqrt(2)
    return S


zero = _build_state(State.ZERO)
one = _build_state(State.ONE)
plus = _build_state(State.PLUS)
minus = _build_state(State.MINUS)
iplus = _build_state(State.iPLUS)
iminus = _build_state(State.iMINUS)
