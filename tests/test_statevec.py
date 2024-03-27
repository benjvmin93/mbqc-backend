import unittest
from src.state_vec import StateVec, meas_op
from src.pauli import Plane, MeasureUpdate
from src.clifford import get
from src.state import zero, one, plus, minus, iplus, iminus
import numpy as np


def ptrace(psi, qargs):
    """Perform partial trace of the selected qubits.
    .. warning::
        This method currently assumes qubits in qargs to be separable from the rest
        (checks not implemented for speed).
        Otherwise, the state returned will be forced to be pure which will result in incorrect output.
        Correct behaviour will be implemented as soon as the densitymatrix class, currently under development
        (PR #64), is merged.
    Parameters
    ----------
    qargs : list of int
        qubit indices to trace over
    """
    nqubit_after = len(psi.shape) - len(qargs)
    rho = np.tensordot(psi, psi.conj(), axes=(qargs, qargs))  # density matrix
    rho = np.reshape(rho, (2**nqubit_after, 2**nqubit_after))
    evals, evecs = np.linalg.eig(rho)  # back to statevector
    new_psi = np.reshape(evecs[:, np.argmax(evals)], (2,) * nqubit_after)
    return new_psi


def init_statevec(nQubits: int = 1) -> StateVec:
    return StateVec(nQubits, [0])


class TestStatevec(unittest.TestCase):
    def test_remove_one_qubit(self):
        """
        Removes one qubit and checks if the state vector is correct.
        """
        nQubits = 2
        sv = StateVec(list(range(nQubits)))

        for i in range(nQubits):
            sv.entangle(i, (i + 1) % nQubits)

        sv2 = np.copy(sv.psi)
        sv.measure(
            index=0,
            plane="XY",
            angle=0,
            s_domain=[],
            t_domain=[],
            measurements=[None, None],
        )

        sv2 = ptrace(sv2, [0])
        sv2 /= np.linalg.norm(sv2)

        np.testing.assert_almost_equal(
            np.abs(sv.psi.flatten().dot(sv2.flatten().conj())), 1
        )

    def test_measurement_into_each_XYZ_basis(self):
        """
        Measure statevec in every plane basis, remove the measured qubit and compare with a fresh statevec.
        """
        n = 3
        k = 0
        for plane in ["XY", "YZ", "XZ"]:
            measure_update = MeasureUpdate.compute(Plane[plane], 0, 0, get(0))
            vec = measure_update.new_plane.polar(0)
            m_op = meas_op(vec, 0)
            sv = StateVec(list(range(n)))
            sv.single_qubit_evolution(m_op, [k])
            sv.remove_qubit(k)
            sv2 = StateVec(list(range(n - 1)))
            np.testing.assert_almost_equal(
                np.abs(sv.psi.flatten() @ sv2.psi.flatten().conj()), 1
            )

    def test_measurement(self):
        """
        Measure statevec into each state projector (modulo |1>)
        """
        n = 3
        k = 0
        for state in [plus, zero, one, iplus, iminus]:
            m_op = np.outer(state, state.T.conjugate())
            sv = StateVec(list(range(n)))
            sv.single_qubit_evolution(m_op, [k])
            sv.remove_qubit(k)
            sv2 = StateVec(list(range(n - 1)))
            np.testing.assert_almost_equal(
                np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1
            )

    def test_measurement_one(self):
        """
        Measure statevec into |-><-|
        """
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        state = minus
        m_op = np.outer(state, state.T.conjugate())
        sv = StateVec(list(range(n)))
        sv.psi = sv.single_qubit_evolution(m_op, [k])
        with self.assertRaises(AssertionError):
            sv.remove_qubit(k)
