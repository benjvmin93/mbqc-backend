from src.state import zero, one, plus, minus, iminus, iplus
import numpy as np
import unittest

class TestState(unittest.TestCase):
    def test_states(self):
        assert np.array_equiv(zero.flatten(), np.array([1, 0], dtype=complex))
        assert np.array_equiv(one.flatten(), np.array([0, 1], dtype=complex))
        assert np.array_equiv(plus.flatten(), np.array([1, 1] / np.sqrt(2), dtype=complex))
        assert np.array_equiv(minus.flatten(), np.array([1, -1] / np.sqrt(2), dtype=complex))
        assert np.array_equiv(iplus.flatten(), np.array([1, 1j] / np.sqrt(2), dtype=complex))
        assert np.array_equiv(iminus.flatten(), np.array([1, -1j] / np.sqrt(2), dtype=complex))