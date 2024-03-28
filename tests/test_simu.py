from graphix.graphix import Circuit
from src.mbqc import MBQC
import numpy as np
import unittest


def fidelity(psi: np.ndarray, phi: np.ndarray) -> bool:
    return np.abs(np.dot(psi.conjugate(), phi)) ** 2


def get_pattern_from_circ(circ):
    pattern = circ.standardize_and_transpile()
    return MBQC(pattern)


class TestSimu(unittest.TestCase):
    def test_H_1(self):
        c = Circuit(1)
        c.h(0)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([1, 0])
        statevec = mbqc.state_vec.get_state_vector()
        assert np.isclose(fidelity(statevec.flatten(), expected.flatten()), 1.0)

    def test_H_2(self):
        c = Circuit(1)
        c.h(0)
        c.h(0)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([1, 1]) / np.sqrt(2)
        statevec = mbqc.state_vec.get_state_vector()
        assert np.isclose(fidelity(statevec.flatten(), expected.flatten()), 1.0)

    def test_cnot(self):
        c = Circuit(2)
        c.cnot(0, 1)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([1, 1, 1, 1]) / 2
        statevec = mbqc.state_vec.get_state_vector()
        assert np.isclose(fidelity(statevec.flatten(), expected.flatten()), 1.0)

    def test_cnot_2(self):
        c = Circuit(2)
        c.h(0)
        c.x(0)
        c.h(1)
        c.cnot(0, 1)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([0, 0, 0, 1])
        statevec = mbqc.state_vec.get_state_vector()
        assert np.isclose(fidelity(statevec.flatten(), expected.flatten()), 1.0)

    def test_y(self):
        c = Circuit(1)
        c.y(0)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([-1j, 1j]) / np.sqrt(2)
        statevec = mbqc.state_vec.get_state_vector()
        assert np.isclose(fidelity(statevec.flatten(), expected.flatten()), 1.0)


if __name__ == "__main__":
    unittest.main()
