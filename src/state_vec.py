import numpy as np
from .clifford import X, Z, get
from . import pauli
from .command import Plane
from .logger import logger
from .state import plus

CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)

SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


def meas_op(
    s_signal: int, t_signal: int, angle: float, plane: Plane, vop: int, measurement: int
) -> np.ndarray:
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
    cliff_op = get(vop)
    measure_update = pauli.MeasureUpdate.compute(
        pauli.Plane[plane], s_signal % 2 == 1, t_signal % 2 == 1, cliff_op
    )
    angle = angle * measure_update.coeff + measure_update.add_term
    angle *= np.pi

    # Build projector operator
    vec = measure_update.new_plane.polar(angle)
    op_mat = np.eye(2, dtype=np.complex128) / 2
    for i in range(3):
        cliff_op = get(i + 1)
        op_mat += (-1) ** (measurement) * vec[i] * cliff_op.matrix / 2
    logger.debug(f"[meas_op]: angle={angle}, mOP=\n{op_mat}")
    return op_mat


class StateVec:
    def __init__(self, nQubits: int = 1):
        """
        Initalize a new state vector in the |+> state.
        """
        self.psi = np.ones((2,) * nQubits) / 2 ** (
            nQubits / 2
        )  # Initialize statevector in |+> ⊗^n
        self.node_index = list(range(nQubits))

    def __repr__(self) -> str:
        return str(f"{self.psi.flatten()}")

    def __eq__(self, __value: object) -> bool:
        return np.array_equal(self.psi.flatten(), np.array(__value))

    @property
    def norm(self) -> float:
        """
        Returns the norm of the state vector.
        """
        return _norm(self.psi)

    def tensor(self, other: np.ndarray):
        """
        Performs self ⊗ other.
        """
        new_shape = len(self.psi.shape) + len(other.shape)
        self.psi = np.kron(self.psi.flatten(), other.flatten()).reshape(
            (2,) * new_shape
        )

    def get_state_vector(self) -> np.ndarray:
        return self.psi

    def prepare_state(self, target: int) -> None:
        """
        Append new qubit to the end of self.psi and update self.node_index
        We assume that 'target' doesn't exist in the state vector because we
        shouldn't prepare inputs qubits nor qubits that have already been prepared
        """
        new_qubit = plus
        self.tensor(new_qubit)
        self.node_index.append(target)
        logger.debug(
            f"[N]({self.node_index.index(target)}): statevec={self.psi.flatten()}, shape={self.psi.shape}"
        )

    def entangle(self, control: int, target: int) -> None:
        """
        Entangles the two qubits by applying CZ on target according to control.
        """
        assert control != target
        # contraction: 2nd index - control index, and 3rd index - target index.
        control = self.node_index.index(control)
        target = self.node_index.index(target)
        self.psi = np.tensordot(CZ_TENSOR, self.psi, ((2, 3), (control, target)))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), (control, target))
        logger.info(f"Entangling qubit {control} with qubit {target}")
        logger.debug(f"[E]({control},{target}): statevec={self.psi.flatten()}")

    def swap(self, qubits: tuple[int, int]) -> None:
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (i, j) qubit indices
        """
        logger.info(f"Swap qubits {qubits[0]} with {qubits[1]}")

        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)
        logger.debug(
            f"[SWAP]({qubits[0]},{qubits[1]}): statevec={self.psi.flatten()}, shape={self.psi.shape}"
        )

    def measure(
        self,
        index: int,
        plane: Plane,
        angle: int,
        s_domain: list[int],
        t_domain: list[int],
        vop: int = 0,
    ) -> int:
        """
        Measure the qubit at index.
        Returns:
            list[int]: The updated measurements list.
        """
        logger.info(f"Measuring qubit {index} in plane {plane} and angle {angle}.")

        ### Build measurement operator
        s_signal = s_domain if isinstance(s_domain, int) else sum(s_domain)
        t_signal = t_domain if isinstance(t_domain, int) else sum(t_domain)
        proj_plus = meas_op(s_signal, t_signal, angle, plane, vop, 0)
        proj_minus = meas_op(s_signal, t_signal, angle, plane, vop, 1)

        # Get right index within self.node_index
        index_sv = self.node_index.index(index)

        # Get projected states
        projected_plus = np.tensordot(proj_plus, self.psi, (1, index_sv))
        projected_plus = np.moveaxis(projected_plus, 0, index_sv)

        # Computes probabilities of getting each state
        proba_Plus = np.linalg.norm(projected_plus) ** 2
        proba_Minus = 1 - proba_Plus

        logger.debug(f"[M]({index}): p(+)={proba_Plus}, p(-)={proba_Minus}")

        # Simulate measurement according to probabilities
        measurement = np.random.choice(a=[0, 1], p=[proba_Plus, proba_Minus])

        if measurement == 0:  # We already computed the state projected over |+>
            self.psi = projected_plus
        else:  # Project onto |->
            self.single_qubit_evolution(proj_minus, index_sv)

        logger.debug(f"[M]({index}): res={measurement}")

        logger.debug(
            f"[M]({index}): projected_state={self.psi.flatten()}, shape={self.psi.flatten().shape}"
        )

        # Remove measured qubit from state vector
        self.remove_qubit(index_sv)
        # Remove qubit index from node list
        self.node_index.remove(index)

        return measurement

    def apply_correction(
        self, type: str, index: int, domain: list[int], measurement_results: list[int]
    ) -> None:
        """
        Applies correction 'X' or 'Z' to the qubit at 'index' according to the signal domain measurements.
        """
        # Ensure qubit indices in domain have already been measured.
        # This should be done in mbqc class.
        for i in domain:
            assert measurement_results[i] != None

        # Get right index within self.node_index
        sv_index = self.node_index.index(index)

        cliff_gate = X if type == "X" else Z
        if np.mod(sum([measurement_results[i] for i in domain]), 2) == 1:
            self.single_qubit_evolution(cliff_gate.matrix, sv_index)
            logger.info(f"[{type}]({index}): new_psi={self.psi.flatten()}")

    def single_qubit_evolution(self, op: np.ndarray, index: int):
        """
        Apply one qubit operator to |psi> at right index.
        """
        self.psi = np.tensordot(op, self.psi, (1, index))
        self.psi = np.moveaxis(self.psi, 0, index)

    def multi_qubit_evolution(self, op: np.ndarray, qargs: tuple[int, int]) -> None:
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

    def normalize(self) -> None:
        """
        Normalize vector state (ie. divides it by its norm).
        """
        self.psi = self.psi / _norm(self.psi)

    def remove_qubit(self, index: int) -> None:
        """
        Remove qubit at 'index' from the state vector.
        """
        assert not np.isclose(_norm(self.psi), 0)
        psi = self.psi.take(indices=0, axis=index)
        self.psi = (
            psi
            if not np.isclose(_norm(psi), 0)
            else self.psi.take(indices=1, axis=index)
        )
        self.normalize()
        logger.debug(
            f"[remove_qubit]: new_psi={self.psi.flatten()}, shape={self.psi.flatten().shape}, norm={_norm(self.psi)}"
        )


def _norm(psi: np.ndarray) -> float:
    """
    Computes the norm of a state vector.
    """
    return np.sqrt(np.sum(psi.flatten().conj() * psi.flatten()))
