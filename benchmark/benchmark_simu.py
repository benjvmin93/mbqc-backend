from copy import deepcopy
import time
from src.mbqc import MBQC
from src.command import N, E, M, X, Z
from src.state_vec import StateVec
from graphix.graphix.simulator import PatternSimulator
from graphix.graphix import Circuit, Pattern
import numpy as np


def get_random_different_from(random_list: list[int], max: int):
    b = np.random.randint(max)
    while b in random_list:
        b = np.random.randint(max)
    return b


def simple_random_circuit(nqubit, depth):
    r"""Generate a test circuit for benchmarking.

    This function generates a circuit with nqubit qubits and depth layers,
    having layers of CNOT and Rz gates with random placements.

    Parameters
    ----------
    nqubit : int
        number of qubits
    depth : int
        number of layers

    Returns
    -------
    circuit : graphix.transpiler.Circuit object
        generated circuit
    """
    qubit_index = [i for i in range(nqubit)]
    circuit = Circuit(nqubit)
    for _ in range(depth):
        np.random.shuffle(qubit_index)
        for j in range(len(qubit_index) // 2):
            circuit.cnot(qubit_index[2 * j], qubit_index[2 * j + 1])
        for j in range(len(qubit_index)):
            circuit.rz(qubit_index[j], 2 * np.pi * np.random.random())
    return circuit


def finalize(pattern, state_vec):
    sort_qubits(pattern, state_vec)
    state_vec.normalize()


def sort_qubits(pattern, state_vec) -> None:
    """sort the qubit order in internal statevector"""
    # print(f'output_nodes: {pattern.output_nodes}')
    for i, ind in enumerate(pattern.output_nodes):
        # print(f'node_index: {state_vec.node_index}')
        if not state_vec.node_index[i] == ind:
            move_from = state_vec.node_index.index(ind)
            state_vec.swap((i, move_from))
            state_vec.node_index[i], state_vec.node_index[move_from] = (
                state_vec.node_index[move_from],
                state_vec.node_index[i],
            )


def g_finalize(pattern, backend):
    g_sort_qubits(pattern, backend)
    backend.state.normalize()


def g_sort_qubits(pattern, backend) -> None:
    """sort the qubit order in internal statevector"""
    for i, ind in enumerate(pattern.output_nodes):
        if not backend.node_index[i] == ind:
            move_from = backend.node_index.index(ind)
            backend.state.swap((i, move_from))
            backend.node_index[i], backend.node_index[move_from] = (
                backend.node_index[move_from],
                backend.node_index[i],
            )


def get_exec_time(fun, args: list = []):
    """
    Get the execution time of a function.
    Returns a tuple with the return of the function and the time it last
    """
    t1 = time.time()
    res = fun(*args)
    t2 = time.time()
    return (res, t2 - t1)


def run_pattern(simu: MBQC, time_dict: dict):
    """
    Our simulator. Updates time_dict with the execution times for each pattern commands.
    """
    cmd_list = simu.pattern.cmd_list
    simu.state_vec = StateVec(simu.pattern.input_nodes)
    for cmd in cmd_list:
        # print(f"Node index: {simu.state_vec.node_index}")
        match cmd:
            case N(node=i):
                time = get_exec_time(simu.state_vec.prepare_state, [i])
                time_dict["N"]["simu"] += time[1]
            case E(nodes=(i, j)):
                time = get_exec_time(simu.state_vec.entangle, [i, j])
                time_dict["E"]["simu"] += time[1]
            case M(node=i, plane=p, angle=alpha, s_domain=s_domain, t_domain=t_domain):
                time = get_exec_time(
                    simu.state_vec.measure,
                    [i, p, alpha, s_domain, t_domain, simu.measurements],
                )
                simu.measurements[i] = time[0]
                time_dict["M"]["simu"] += time[1]
            case X(node=i, domain=domain):
                time = get_exec_time(
                    simu.state_vec.apply_correction, ["X", i, domain, simu.measurements]
                )
                time_dict["X"]["simu"] += time[1]
            case Z(node=i, domain=domain):
                time = get_exec_time(
                    simu.state_vec.apply_correction, ["Z", i, domain, simu.measurements]
                )
                time_dict["Z"]["simu"] += time[1]
            case _:
                e = f"Command type {cmd} doesn't exist."
                raise KeyError(e)
    time = get_exec_time(finalize, [simu.pattern, simu.state_vec])
    time_dict["finalize"]["simu"] += time[1]


def g_run_pattern(sv_simu: PatternSimulator, time_dict: dict):
    """
    Graphix run pattern function override.
    Updates time_dict with execution times according to each commands.
    """
    pattern = sv_simu.pattern
    backend = sv_simu.backend
    backend.add_nodes(pattern.input_nodes)
    for cmd in list(pattern):
        if cmd[0] == "N":
            time = get_exec_time(backend.add_nodes, [[cmd[1]]])
            time_dict["N"]["graphix"] += time[1]
        elif cmd[0] == "E":
            time = get_exec_time(backend.entangle_nodes, [cmd[1]])
            time_dict["E"]["graphix"] += time[1]
        elif cmd[0] == "M":
            time = get_exec_time(backend.measure, [cmd])
            time_dict["M"]["graphix"] += time[1]
        elif cmd[0] == "X":
            time = get_exec_time(backend.correct_byproduct, [cmd])
            time_dict["X"]["graphix"] += time[1]
        elif cmd[0] == "Z":
            time = get_exec_time(backend.correct_byproduct, [cmd])
            time_dict["Z"]["graphix"] += time[1]
        elif cmd[0] == "C":
            backend.apply_clifford(cmd)
        else:
            raise ValueError("invalid commands")
    time = get_exec_time(g_finalize, [sv_simu.pattern, sv_simu.backend])
    time_dict["finalize"]["graphix"] += time[1]


class BenchmarkSimu:
    """
    Benchmark class used to compare graphix state vector simulator and our simulator.
    """

    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        self.__sv_simu = MBQC(pattern)
        self.__graphix_simu = PatternSimulator(pattern, pr_calc=True)

    def bench_mbqc_simu(self, it=1000):
        """
        Computes execution times between mbqc simulator with graphix.
        """
        time_dict = {"sv_simu": 0.0, "graphix_simu": 0.0}
        # Run our simulator and graphix simulator multiple times so we can get the average execution times for both.
        for _ in range(it):
            # Copy sv_simu
            sv_copy = deepcopy(self.__sv_simu)
            sv_time = get_exec_time(sv_copy.run_pattern)
            time_dict["sv_simu"] += sv_time[1]
            # Copy graphix simu
            g_sv_copy = deepcopy(self.__graphix_simu)
            graphix_time = get_exec_time(g_sv_copy.run)
            time_dict["graphix_simu"] += graphix_time[1]
            # Ensure we get the same results
            try:
                assert np.array_equal(
                    np.abs(sv_copy.state_vec.psi), np.abs(g_sv_copy.backend.state.psi)
                )
            except:
                print(f"sv: {sv_copy.state_vec.psi.flatten()}")
                print(f"graphix_sv: {g_sv_copy.backend.state.psi.flatten()}")
                break

        time_dict["sv_simu"] /= it
        time_dict["graphix_simu"] /= it

        return time_dict

    def bench_init_times(self, it=1000):
        """
        Benchmark initialization times between graphix simulator and our own simulator.
        """
        init_times = {"simu": 0.0, "graphix": 0.0}
        for _ in range(it):
            t1 = time.time()
            MBQC(self.pattern)
            t2 = time.time()
            init_times["simu"] += t2 - t1
            t1 = time.time()
            PatternSimulator(self.pattern)
            t2 = time.time()
            init_times["graphix"] += t2 - t1

        init_times["simu"] /= it
        init_times["graphix"] /= it

        return init_times

    def bench_cmd_times(self, it=1000):
        """
        Benchmark individual command times. Computes the average of execution time for each commands + finalize method.
        """
        time_cmd_dict = {
            "N": {"simu": 0, "graphix": 0},
            "E": {"simu": 0, "graphix": 0},
            "M": {"simu": 0, "graphix": 0},
            "X": {"simu": 0, "graphix": 0},
            "Z": {"simu": 0, "graphix": 0},
            "finalize": {"simu": 0, "graphix": 0},
        }

        for _ in range(it):
            # Copy sv_simu
            sv_copy = deepcopy(self.__sv_simu)
            run_pattern(sv_copy, time_cmd_dict)
            # Copy graphix simu
            g_sv_copy = deepcopy(self.__graphix_simu)
            g_run_pattern(g_sv_copy, time_cmd_dict)

        # Normalize to get the time average.
        avg_time_cmd = {
            key: {
                "simu": time_cmd_dict[key]["simu"] / it,
                "graphix": time_cmd_dict[key]["graphix"] / it,
            }
            for key in time_cmd_dict.keys()
        }

        return avg_time_cmd
