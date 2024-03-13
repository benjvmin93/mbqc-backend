import numpy as np
from gates import CLIFFORD_GATES, CLIFFORD_CONJ, CLIFFORD_MUL, CLIFFORD_LABEL

def build_unitary_gate(nb_qubits, index, gate):
    """
    Build a unitary gate given the total number of qubits and the qubit index on which to apply the gate passed in argument.
    """
    U = 1
    for i in range(nb_qubits):
        if i == index:
            U = np.kron(U, gate)
        else:
            U = np.kron(U, np.identity(2))
    return U

def build_mop(pstate, nb_qubits, index, alpha):
    print(f"BUILDING MOP {pstate} projecting qubit {index} with alpha={alpha}, nbqubits={nb_qubits}")
    state = str_state_to_array(pstate, alpha)
    M = 1
    for i in range(nb_qubits):
        if i == index:
            M = np.kron(M, state @ state.conjugate().T)
        else:
            M = np.kron(M, np.identity(2))
    return M

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

def meas_op(angle, vop=0, plane="XY", choice=0):
    """Returns the projection operator for given measurement angle and local Clifford op (VOP).

    .. seealso:: :mod:`graphix.clifford`

    Parameters
    ----------
    angle : float
        original measurement angle in radian
    vop : int
        index of local Clifford (vop), see graphq.clifford.CLIFFORD
    plane : 'XY', 'YZ' or 'ZX'
        measurement plane on which angle shall be defined
    choice : 0 or 1
        choice of measurement outcome. measured eigenvalue would be (-1)**choice.

    Returns
    -------
    op : numpy array
        projection operator

    """
    print(f"Build measOP with angle {angle}, clifford: {vop}, plane={plane}, measured={choice}")
    assert vop in np.arange(24)
    assert choice in [0, 1]
    assert plane in ["XY", "YZ", "XZ"]
    if plane == "XY":
        vec = (np.cos(angle), np.sin(angle), 0)
    elif plane == "YZ":
        vec = (0, np.cos(angle), np.sin(angle))
    elif plane == "XZ":
        vec = (np.cos(angle), 0, np.sin(angle))
    op_mat = np.eye(2, dtype=np.complex128) / 2
    for i in range(3):
        op_mat += (-1) ** (choice) * vec[i] * CLIFFORD_GATES[CLIFFORD_LABEL[i + 1]] / 2
    op_mat = CLIFFORD_GATES[CLIFFORD_LABEL[CLIFFORD_CONJ[vop]]] @ op_mat @ CLIFFORD_GATES[CLIFFORD_LABEL[vop]]
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
        return np.array_equal(np.array(self.psi).flatten(), __value)
    
    def get_state_vector(self):
        return self.psi

    def prepare_state(self, qubit_index):
        """
        Prepare |+> state at the right qubit_index within the vector state.
        """
        self.single_qubit_evolution(CLIFFORD_GATES['H'], qubit_index)

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

    def swap(self, qubits):
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)

    def measure(self, index, plane='XY', angle=0, s_domain=[], t_domain=[], measurements=[], vop=0):
        """
        Measure the qubit at index.
        Returns:
            int: The measurement result.
        """

        angle *= np.pi
        print(f"PSI BEFORE MEAS: {self.psi.flatten()}")
        print(f"NODES = {self.node_index}")
        # Get measurement projectors
        Mop_plus = build_mop('|+>', self.nb_qubits, self.node_index.index(index), angle * np.pi)
        Mop_minus = build_mop('|->', self.nb_qubits, self.node_index.index(index), angle * np.pi)

        # Reshape our state vector so we can multiply it with measurement operators
        state_vec = self.psi.flatten().reshape(2 ** self.nb_qubits)

        # Computes probabilities of getting each state
        proba_Plus = np.dot(state_vec.conjugate().T @ Mop_plus.conjugate().T, Mop_plus @ state_vec)
        proba_Minus = np.dot(state_vec.conjugate().T @ Mop_minus.conjugate().T, Mop_minus @ state_vec)

        print(f"NORM = {_norm(self.psi)}")
        print(f"Proba + = {proba_Plus}, proba - = {proba_Minus}")

        # Simulate measurement according to probabilities
        measurement = np.random.choice(a=[0, 1], p=[np.abs(proba_Plus), np.abs(proba_Minus)])        
        
        mop = None
        if measurement == 0:
            # Collapse in state |+>
            mop = str_state_to_array('|+>', angle * np.pi)
        else:
            # Collapse in state |->
            mop = str_state_to_array('|->', angle * np.pi)

        if np.sum(s_domain) % 2 == 1:
            vop = CLIFFORD_MUL[1, vop]
        if np.sum(t_domain) % 2 == 1:
            vop = CLIFFORD_MUL[3, vop]
        
        print(f"MEASURED {measurement} on qubit {index}")
        mop = meas_op(angle, vop=vop, plane=plane, choice=measurement)
        
        #mop = mop @ mop.conjugate().T

        print(f"====== MOP =\n{mop}========\n")

        self.single_qubit_evolution(mop, self.node_index.index(index))
        self.psi[np.abs(self.psi) < 1e-15] = 0
        print(f"PSI AFTER MEAS = {self}")
        self.remove_qubit(self.node_index.index(index))
        self.node_index.remove(index)
        self.nb_qubits -= 1
        measurements[index] = measurement
        return measurements

    def apply_correction(self, type, index, domain, measurement_results):
        for i in domain: assert measurement_results[i] != None
        index = self.node_index.index(index)
        if np.mod(sum([measurement_results[i] for i in domain]), 2) == 1:
            print(f"Applying {type} on qubit {index}")
            self.single_qubit_evolution(CLIFFORD_GATES[type], index)
            print(self.psi)

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
        print(f"NEW PSI AFTER QUBIT {self.node_index[index]} REMOVED:\n{self.psi.flatten()}")
    
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
