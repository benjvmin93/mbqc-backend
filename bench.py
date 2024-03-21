from benchmark.benchmark_simu import BenchmarkSimu
from graphix.graphix import Circuit
import numpy as np
from src.state_vec import StateVec
from src.mbqc import MBQC
from src.command import N, M, E, X, Z


def get_pattern(circuit):
    return circuit.standardize_and_transpile()


def get_random_different_from(random_list: list[int], max: int):
    b = np.random.randint(max)
    while b in random_list:
        b = np.random.randint(max)
    return b


def build_random_circuit(n: int = 1, depth: int = 1):
    """
    Build a random circuit according to the number of qubits and depth of the circuit.
    Picks random between [0,3[ to choose which type of gate.
    Then picks appropriate control (if there are) and target, which are ensured to be different thanks to get_random_different_from function.
    """
    circuit = Circuit(n)
    one_qubit_gates = [
        circuit.h,
        circuit.s,
        circuit.x,
        circuit.y,
        circuit.z,
        circuit.rx,
        circuit.ry,
        circuit.rz,
    ]
    two_qubit_gates = [circuit.cnot, circuit.swap, circuit.rzz]
    while depth != 0:
        gate_type = 0  # If there is one qubit, we can only use one_qubit gates
        if n > 1:
            gate_type = np.random.randint(2)

        if gate_type == 0:  # One qubit gate
            gate = one_qubit_gates[np.random.randint(len(one_qubit_gates))]
            target = np.random.randint(n)
            if gate == circuit.rx or gate == circuit.ry or gate == circuit.rz:
                gate(target, np.random.rand() * np.pi)
            else:
                gate(target)
        elif gate_type == 1:  # Two qubits gate
            gate = two_qubit_gates[np.random.randint(len(two_qubit_gates))]
            control = np.random.randint(n)
            target = get_random_different_from([control], n)
            if gate == circuit.rzz:
                gate(control, target, np.random.rand() * np.pi)
            else:
                gate(control, target)
        depth -= 1
    return circuit


def bench_sv_simu(circuit: Circuit, it=1000):
    """
    Compare statevec with graphix simulator according to the given circuit.
    """
    p = get_pattern(circuit)

    def run_our_simulator():
        simu = MBQC(list(p), p.Nnode, p.input_nodes, p.output_nodes)
        simu.run_pattern()

    def run_graphix_simulator():
        p.simulate_pattern(pr_calc=True)

    bench_simu = BenchmarkSimu().bench_functions(
        functions=(run_our_simulator, run_graphix_simulator),
        label1="simu",
        label2="graphix-simu",
        it=it,
    )
    return bench_simu


from graphix.graphix import Circuit, Pattern
from graphix.graphix.sim.statevec import StatevectorBackend
import time


def finalize(pattern, state_vec):
    sort_qubits(pattern, state_vec)
    state_vec.normalize()


def sort_qubits(pattern, state_vec) -> None:
    """sort the qubit order in internal statevector"""
    for i, ind in enumerate(pattern.output_nodes):
        if not state_vec.node_index[i] == ind:
            move_from = state_vec.node_index.index(ind)
            state_vec.swap((i, move_from))
            state_vec.node_index[i], state_vec.node_index[move_from] = (
                state_vec.node_index[move_from],
                state_vec.node_index[i],
            )


def run_pattern(simu: MBQC, time_dict: dict):
    """
    Our simulator. Updates time_dict with the execution times for each pattern commands.
    """
    state_vec = simu.state_vec
    cmd_list = simu.pattern.cmd_list
    for cmd in cmd_list:
        match cmd:
            case N(node=i):
                t1 = time.time()
                state_vec.prepare_state(i)
                t2 = time.time()
                time_dict["N"]["simu"] += t2 - t1
            case E(nodes=(i, j)):
                t1 = time.time()
                state_vec.entangle(i, j)
                t2 = time.time()
                time_dict["E"]["simu"] += t2 - t1
            case M(node=i, plane=p, angle=alpha, s_domain=s_domain, t_domain=t_domain):
                t1 = time.time()
                simu.measurements[i] = state_vec.measure(
                    i, p, alpha, s_domain, t_domain, simu.measurements
                )
                t2 = time.time()
                time_dict["M"]["simu"] += t2 - t1
            case X(node=i, domain=domain):
                t1 = time.time()
                state_vec.apply_correction("X", i, domain, simu.measurements)
                t2 = time.time()
                time_dict["X"]["simu"] += t2 - t1
            case Z(node=i, domain=domain):
                t1 = time.time()
                state_vec.apply_correction("Z", i, domain, simu.measurements)
                t2 = time.time()
                time_dict["Z"]["simu"] += t2 - t1
            case _:
                e = f"Command type {cmd} doesn't exist."
                raise KeyError(e)
    finalize(simu.pattern, simu.state_vec)


def g_run_pattern(sv_simu: StatevectorBackend, time_dict: dict):
    """
    Graphix run pattern function override.
    Updates time_dict with execution times according to each commands.
    """
    pattern = sv_simu.pattern
    sv_simu.add_nodes(pattern.input_nodes)
    for cmd in list(pattern):
        if cmd[0] == "N":
            t1 = time.time()
            sv_simu.add_nodes([cmd[1]])
            t2 = time.time()
            time_dict["N"]["graphix"] += t2 - t1
        elif cmd[0] == "E":
            t1 = time.time()
            sv_simu.entangle_nodes(cmd[1])
            t2 = time.time()
            time_dict["E"]["graphix"] += t2 - t1
        elif cmd[0] == "M":
            t1 = time.time()
            sv_simu.measure(cmd)
            t2 = time.time()
            time_dict["M"]["graphix"] += t2 - t1
        elif cmd[0] == "X":
            t1 = time.time()
            sv_simu.correct_byproduct(cmd)
            t2 = time.time()
            time_dict["X"]["graphix"] += t2 - t1
        elif cmd[0] == "Z":
            t1 = time.time()
            sv_simu.correct_byproduct(cmd)
            t2 = time.time()
            time_dict["Z"]["graphix"] += t2 - t1
        elif cmd[0] == "C":
            sv_simu.apply_clifford(cmd)
        else:
            raise ValueError("invalid commands")
    sv_simu.finalize()


def bench_run_pattern(circuit, it=1000):
    """
    Benchmark pattern command execution times.
    Returns every average commands execution times according to our simulator and the graphix one.
    Additionally, returns the total average execution times of the pattern for both simulators.
    """
    time_dict = {
        "N": {"simu": 0, "graphix": 0},
        "E": {"simu": 0, "graphix": 0},
        "M": {"simu": 0, "graphix": 0},
        "X": {"simu": 0, "graphix": 0},
        "Z": {"simu": 0, "graphix": 0},
    }

    # Get pattern
    p = Circuit.standardize_and_transpile(circuit)

    # Run multiple iterations for our simulator
    tot_simu = 0
    tot_g_simu = 0
    for _ in range(it):
        simu = MBQC(list(p), p.Nnode, p.input_nodes, p.output_nodes)
        g_simu = StatevectorBackend(p, pr_calc=True)  # Enable probability calculation

        t1 = time.time()
        run_pattern(simu, time_dict)
        t2 = time.time()
        # Get total exec time
        tot_simu += t2 - t1
        t1 = time.time()
        g_run_pattern(g_simu, time_dict)
        t2 = time.time()
        tot_g_simu += t2 - t1

    # Compute average time
    tot_simu /= it
    tot_g_simu /= it

    # Compute average time for each command
    for key in time_dict.keys():
        time_dict[key]["simu"] /= it
        time_dict[key]["graphix"] /= it

    return (time_dict, {"tot_simu": tot_simu, "tot_g_simu": tot_g_simu})


# nQubits = 2
# circ = Circuit(nQubits)
# circ.h(0)
# circ.cnot(0, 1)
# node = 0
# angle = 0
# vop = 0
#
# b_tensor = bench_sv_tensor(circ, 1)
# print(f"bench tensor: {b_tensor}")
# b_measure = bench_sv_measure(circ, node, angle, vop=vop)
# print(f"bench measure: {b_measure}")
# b_meas_op = bench_meas_op(angle, vop, "XY", 0, 0, 0)
# print(f"bench meas_op: {b_meas_op}")
#
# type = "X"
# domain = [1]
# node = 0
# measurements = [None, 1, None, None]
# b_apply_correction = bench_apply_correction(circ, type, node, domain, measurements)
# print(f"bench apply_correction: {b_apply_correction}")


def bench_n_depth(depth=1):
    perf_over_depth = {}
    for i in range(1, depth + 1):
        circ = build_random_circuit(1, i)
        (tot_simu, tot_g_simu) = bench_run_pattern(circ)
        perf_over_depth[i] = {"simu": tot_simu, "g-simu": tot_g_simu}
    return perf_over_depth


def bench_n_qubits(n_qubits=1):
    perf_over_n_qubits = {}
    for i in range(1, n_qubits + 1):
        circ = build_random_circuit(i, 1)
        (tot_simu, tot_g_simu) = bench_run_pattern(circ)
        perf_over_n_qubits[i] = {"simu": tot_simu, "g-simu": tot_g_simu}
    return perf_over_n_qubits
