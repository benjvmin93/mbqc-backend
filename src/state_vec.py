import numpy as np
from .clifford import *
from .pauli import *
from .logger import logger

def str_state_to_array(str_state, alpha=0):
    state_to_array = {
        '|0>': np.array([1, 0]),
        '|1>': np.array([[0], [1]]),
        '|+>': np.array([[1], [np.exp(1j * alpha)]]) / np.sqrt(2),
        '|->': np.array([[1], [-np.exp(1j * alpha)]]) / np.sqrt(2)
    }
    return state_to_array[str_state]

CZ_TENSOR = np.array(
        [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
        dtype=np.complex128,
    )

SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)

def meas_op(s_signal, t_signal, angle, plane, vop, measurement):
    """Returns the projection operator.

    Parameters
    ----------
    s_signal: int
        sum of all s_domain indices
    t_signal: int
        sum of all t_domain indices
    angle : float
        original measurement angle in radian
    plane : 'XY', 'YZ' or 'ZX'
        measurement plane on which angle shall be defined
    vop : int
        index of local Clifford (vop), see clifford.TABLE
    measurement : 0 or 1
        choice of measurement outcome. measured eigenvalue would be (-1)**choice.

    Returns
    -------
    op : numpy array
        projection operator

    """
    # Update angle
    measure_update = pauli.MeasureUpdate.compute(
            pauli.Plane[plane], s_signal % 2 == 1, t_signal % 2 == 1, clifford.TABLE[vop]
        )
    angle = angle * measure_update.coeff + measure_update.add_term
    angle *= np.pi
    # Build projector operator
    vec = measure_update.new_plane.polar(angle)
    op_mat = np.eye(2, dtype=np.complex128) / 2
    for i in range(3):
        op_mat += (-1) ** (measurement) * vec[i] * clifford.CLIFFORD[i + 1] / 2
    logger.debug(f'[meas_op]: angle={angle}, mOP=\n{op_mat}')
    return op_mat

class StateVec:
    def __init__(self, pattern):
        """
        Initalize a new state vector of n qubits in the state \otimes^n |0>
        """
        self.pattern = pattern
        self.nb_qubits = pattern.Nnode
        self.psi = np.zeros((2,) * self.nb_qubits)
        self.psi[(0,) * self.nb_qubits] = 1
        self.node_index = list(range(self.nb_qubits))
    
    def __repr__(self) -> str:
        return str(f"{self.psi.flatten()}")
    
    def __eq__(self, __value: object) -> bool:
        return np.array_equal(self.psi.flatten(), __value)
    
    def get_state_vector(self):
        return self.psi

    def prepare_state(self, qubit_index):
        """
        Prepare |+> state at the right qubit_index within the vector state.
        """
        self.single_qubit_evolution(clifford.CLIFFORD[clifford.CLIFFORD_LABEL.index('H')], qubit_index)
        logger.info(f'Preparing qubit {qubit_index}.')
        logger.debug(f'[N](0): statevec={self.psi.flatten()}')

    def entangle(self, control, target):
        """
        Entangles the two qubits at indices[0] and indices[1]
        """
        assert control != target
        # contraction: 2nd index - control index, and 3rd index - target index.
        control = self.node_index.index(control)
        target = self.node_index.index(target)
        self.psi = np.tensordot(CZ_TENSOR, self.psi, ((2, 3), (control, target)))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))
        logger.info(f'Entangling qubit {control} with qubit {target}')
        logger.debug(f'[E]({control},{target}): statevec={self.psi.flatten()}, shape={self.psi.shape}')

    def swap(self, qubits):
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (i, j) qubit indices
        """
        logger.info(f'Swap qubits {qubits[0]} with {qubits[1]}')
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)
        logger.debug(f'[SWAP]({qubits[0]},{qubits[1]}): statevec={self.psi.flatten()}, shape={self.psi.shape}')


    def measure(self, index, plane='XY', angle=0, s_domain=[], t_domain=[], measurements=[], vop=0):
        """
        Measure the qubit at index.
        Returns:
            list[int]: The updated measurements list.
        """
        logger.info(f'Measuring qubit {index} in plane {plane} and angle {angle}.')     
        
        ### Build measurement operator
        s_signal = sum(s_domain)
        t_signal = sum(t_domain)
        proj_plus = meas_op(s_signal, t_signal, angle, plane, vop, 0)
        proj_minus = meas_op(s_signal, t_signal, angle, plane, vop, 1)

        # Get projected states
        index_in_vect_state = self.node_index.index(index)
        projected_plus = np.tensordot(proj_plus, self.psi, (1, index_in_vect_state))
        projected_plus = np.moveaxis(projected_plus, 0, index_in_vect_state).flatten()
        projected_minus = np.tensordot(proj_minus, self.psi, (1, index_in_vect_state))
        projected_minus = np.moveaxis(projected_minus, 0, index_in_vect_state).flatten()

        """
        # Get projected state transconjugates
        projected_plus_transconj = np.tensordot(self.psi.conj().T, proj_plus.conj().T, (index_in_vect_state, 1))
        projected_plus_transconj = np.moveaxis(projected_plus_transconj, index_in_vect_state, 0)
        projected_minus_transconj = np.tensordot(self.psi.conj().T, proj_minus.conj().T, (index_in_vect_state, 1))
        projected_minus_transconj = np.moveaxis(projected_minus_transconj, index_in_vect_state, 0)
        """
        logger.debug(f'[M]({index}): projected_plus={projected_plus.flatten()}, projected_minus={projected_minus.flatten()}')
        # Computes probabilities of getting each state
        shape = (1, 2 ** self.nb_qubits)
        shape_transconj = (2 ** self.nb_qubits, 1)
        proba_Plus = np.linalg.norm(projected_plus) ** 2
        proba_Minus = np.linalg.norm(projected_minus) ** 2

        logger.debug(f'[M]({index}): p(+)={proba_Plus}, p(-)={proba_Minus}')
        # Simulate measurement according to probabilities and get right measurement operator
        measurement = np.random.choice(a=[0, 1], p=[np.abs(proba_Plus), np.abs(proba_Minus)])
        measurements[index] = measurement
        mop = proj_plus if measurement == 0 else proj_minus

        logger.debug(f'[M]({index}): res={measurement},\nmOP=\n{mop}')

        # Project the state
        self.single_qubit_evolution(mop, index_in_vect_state)
        # self.psi[np.abs(self.psi) < 1e-15] = 0
        logger.debug(f'[M]({index}): projected_state={self.psi.flatten()}, shape={self.psi.flatten().shape}')
        
        # Remove measured qubit
        self.remove_qubit(index_in_vect_state)
        self.node_index.remove(index)

        return measurements

    def apply_correction(self, type, index, domain, measurement_results):
        for i in domain: assert measurement_results[i] != None
        index = self.node_index.index(index)
        if np.mod(sum([measurement_results[i] for i in domain]), 2) == 1:
            self.single_qubit_evolution(clifford.CLIFFORD[clifford.CLIFFORD_LABEL.index(type)], index)
            logger.info(f'[{type}]({index}): new_psi={self.psi.flatten()}')

    def single_qubit_evolution(self, op, index):
        """
        Apply one qubit operator to |psi> at right index.
        """
        self.psi = np.tensordot(op, self.psi, (1, index))
        self.psi = np.moveaxis(self.psi, 0, index)

    def multi_qubit_evolution(self, op, qargs):
        """
        Apply multi qubit operator to |psi> with (control, target) in qargs.
        """
        op_dim = int(np.log2(len(op)))
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        self.psi = np.tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), tuple(qargs)),
        )
        self.psi = np.moveaxis(self.psi, [i for i in range(len(qargs))], qargs)
    
    def normalize(self, threshold=1e-15):
        """
        Normalize vector state (ie. divides it by its norm) and rounds float approximation lower than threshold.
        """
        # self.psi[np.abs(self.psi) < threshold] = 0
        self.psi = self.psi / _norm(self.psi)

    def remove_qubit(self, index):
        assert not np.isclose(_norm(self.psi), 0)
        psi = self.psi.take(indices=0, axis=index)
        self.psi = psi if not np.isclose(_norm(psi), 0) else self.psi.take(indices=1, axis=index)
        self.normalize()
        self.nb_qubits -= 1
        logger.debug(f'[remove_qubit]: new_psi={self.psi.flatten()}, shape={self.psi.flatten().shape}, norm={_norm(self.psi)}')
    
    def sort_qubits(self):
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(self.pattern.output_nodes):
            if not self.node_index[i] == ind:
                move_from = self.node_index.index(ind)
                self.swap((i, move_from))
                self.node_index[i], self.node_index[move_from] = (
                    self.node_index[move_from],
                    self.node_index[i],
                )

    def finalize(self):
        """to be run at the end of pattern simulation."""
        self.sort_qubits()
        self.normalize()

def _norm(psi):
    return np.sqrt(np.sum(psi.flatten().conj() * psi.flatten()))
