from .state_vec import StateVec
from .pattern import Pattern
from .command import N, M, E, X, Z
from .logger import logger


class MBQC:
    def __init__(
        self,
        pattern: Pattern,
    ):
        self.pattern = Pattern(pattern)
        self.measurements = [None] * pattern.Nnode
        self.state_vec = None

    def __repr__(self) -> str:
        return f"statevec: {self.state_vec}, measurements: {self.measurements}"

    def finalize(self):
        self.sort_qubits()
        self.state_vec.normalize()

    def sort_qubits(self) -> None:
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(self.pattern.output_nodes):
            if not self.state_vec.node_index[i] == ind:
                move_from = self.state_vec.node_index.index(ind)
                self.state_vec.swap((i, move_from))
                self.state_vec.node_index[i], self.state_vec.node_index[move_from] = (
                    self.state_vec.node_index[move_from],
                    self.state_vec.node_index[i],
                )

    def run_pattern(self):
        # Initialize statevec with input nodes.
        self.state_vec = StateVec(self.pattern.input_nodes)
        for cmd in self.pattern.cmd_list:
            match cmd:
                case N(node=i):  # Add node in |+>
                    self.state_vec.prepare_state(i)
                case E(nodes=(i, j)):  # Entangle nodes
                    self.state_vec.entangle(i, j)
                case M(  # Measure node
                    node=i, plane=p, angle=alpha, s_domain=s_domain, t_domain=t_domain
                ):
                    self.measurements[i] = self.state_vec.measure(
                        i, p, alpha, s_domain, t_domain, self.measurements
                    )
                case X(node=i, domain=domain):  # Correction X
                    self.state_vec.apply_correction("X", i, domain, self.measurements)
                case Z(node=i, domain=domain):  # Correction Z
                    self.state_vec.apply_correction("Z", i, domain, self.measurements)
                case _:
                    e = f"Command type {cmd} doesn't exist."
                    logger.exception(e)
                    raise KeyError(e)
        self.finalize()
        return self.state_vec
