import unittest
from src.state_vec import StateVec, meas_op
from src.pauli import Plane
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
        sv = StateVec(nQubits, output_nodes=[1])

        for i in range(sv.nb_qubits):
            sv.entangle(i, (i + 1) % sv.nb_qubits)
        m_op = meas_op(
            s_signal=0, t_signal=0, angle=0, plane="XY", vop=0, measurement=0
        )
        sv.single_qubit_evolution(m_op, 0)
        sv2 = np.copy(sv.psi)

        sv.remove_qubit(0)
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
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        for plane in ["XY", "YZ", "XZ"]:
            m_op = meas_op(0, 0, 0, plane, 0, 0)
            sv = StateVec(n, [0])
            sv.single_qubit_evolution(m_op, [k])
            sv.remove_qubit(k)
            sv2 = StateVec(n - 1, [0])
            np.testing.assert_almost_equal(
                np.abs(sv.psi.flatten() @ sv2.psi.flatten().conj()), 1
            )

    def test_measurement(self):
        """
        Measure statevec into each state projector (modulo |1>)
        """
        n = 3
        k = 0
        for state in [plus, zero, minus, iplus, iminus]:
            m_op = np.outer(state, state.T.conjugate())
            sv = StateVec(n, [0])
            sv.single_qubit_evolution(m_op, [k])
            sv.remove_qubit(k)
            sv2 = StateVec(n - 1, [0])
            np.testing.assert_almost_equal(
                np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1
            )

    def test_measurement_one(self):
        """
        Measure statevec into |1><1|
        """
        n = 3
        k = 0
        # for measurement into |1> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        state = one
        m_op = np.outer(state, state.T.conjugate())
        sv = StateVec(n, [0])
        sv.single_qubit_evolution(m_op, [k])
        with self.assertRaises(AssertionError):
            sv.remove_qubit(k)
