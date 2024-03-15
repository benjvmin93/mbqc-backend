from graphix import Circuit
from src.mbqc import MBQC
import numpy as np
import unittest


def get_pattern_from_circ(circ):
    pattern = circ.standardize_and_transpile()
    return MBQC(pattern.seq, pattern.Nnode, pattern.output_nodes)


class TestSimu(unittest.TestCase):
    def test_H_1(self):
        c = Circuit(1)
        c.h(0)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([1, 0])
        statevec = mbqc.state_vec.get_state_vector()
        assert np.allclose(statevec, expected)

    def test_H_2(self):
        c = Circuit(1)
        c.h(0)
        c.h(0)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([1, 1]) / np.sqrt(2)
        statevec = mbqc.state_vec.get_state_vector()
        assert np.allclose(statevec, expected) or np.allclose(statevec, -expected)

    def test_cnot(self):
        c = Circuit(2)
        c.cnot(0, 1)
        mbqc = get_pattern_from_circ(c)
        mbqc.run_pattern()
        expected = np.array([1, 1, 1, 1]) / 2
        statevec = mbqc.state_vec.get_state_vector()
        assert np.allclose(statevec.flatten(), expected) or np.allclose(
            statevec.flatten(), -expected
        )

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
        assert np.allclose(statevec.flatten(), expected) or np.allclose(
            statevec.flatten(), -expected
        )


if __name__ == "__main__":
    unittest.main()
