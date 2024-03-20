from .state_vec import StateVec
from .pattern import Pattern
from .command import N, M, E, X, Z
from .logger import logger


class MBQC:
    def __init__(
        self,
        pattern: list[list],
        Nnode: int,
        input_nodes: list[int],
        output_nodes: list[int],
    ):
        self.pattern = Pattern(pattern, Nnode, input_nodes, output_nodes)
        self.measurements = [None] * Nnode
        self.state_vec = StateVec(
            len(input_nodes)
        )  # Initializes a statevec according to the input nodes.
        logger.info(f"Initialized simulator with {len(input_nodes)} qubits")
        logger.debug(f"Initial statevec = {self.state_vec}, inputs={input_nodes}")

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
        cmd_list = self.pattern.cmd_list
        for cmd in cmd_list:
            match cmd:
                case N(node=i):
                    self.state_vec.prepare_state(i)
                case E(nodes=(i, j)):
                    self.state_vec.entangle(i, j)
                case M(
                    node=i, plane=p, angle=alpha, s_domain=s_domain, t_domain=t_domain
                ):
                    self.measurements[i] = self.state_vec.measure(
                        i, p, alpha, s_domain, t_domain
                    )
                case X(node=i, domain=domain):
                    self.state_vec.apply_correction("X", i, domain, self.measurements)
                case Z(node=i, domain=domain):
                    self.state_vec.apply_correction("Z", i, domain, self.measurements)
                case _:
                    e = f"Command type {cmd} doesn't exist."
                    logger.exception(e)
                    raise KeyError(e)
        self.finalize()
