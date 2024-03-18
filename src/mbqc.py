from .state_vec import StateVec
from .pattern import Pattern
from .command import N, M, E, X, Z
from .logger import logger


class MBQC:
    def __init__(self, pattern: list[list], Nnode: int, input_nodes: list[int], output_nodes: list[int]):
        self.pattern = Pattern(pattern, Nnode, input_nodes, output_nodes)
        self.measurements = [None] * Nnode
        self.state_vec = StateVec(nQubits=1, output_nodes=self.pattern.output_nodes)
        logger.info(f"Initialized simulator with {Nnode} qubits")
        logger.debug(f"Initial statevec = {self.state_vec}")

    def __repr__(self) -> str:
        return f"statevec: {self.state_vec}, measurements: {self.measurements}"

    def run_pattern(self):
        cmd_list = self.pattern.cmd_list
        for i in self.pattern.input_nodes:
            self.state_vec.prepare_state(i)
        for cmd in cmd_list:
            match cmd:
                case N(node=i):
                    self.state_vec.prepare_state(i)
                case E(nodes=(c, t)):
                    self.state_vec.entangle(c, t)
                case M(
                    node=i, plane=p, angle=alpha, s_domain=s_domain, t_domain=t_domain
                ):
                    self.measurements = self.state_vec.measure(
                        i, p, alpha, s_domain, t_domain, self.measurements
                    )
                case X(node=i, domain=domain):
                    self.state_vec.apply_correction("X", i, domain, self.measurements)
                case Z(node=i, domain=domain):
                    self.state_vec.apply_correction("Z", i, domain, self.measurements)
                case _:
                    e = f"Command type {cmd} doesn't exist."
                    logger.exception(e)
                    raise KeyError(e)
        self.state_vec.finalize()
