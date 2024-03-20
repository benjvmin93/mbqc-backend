from benchmark.benchmark_simu import BenchmarkSimu
from src.state_vec import StateVec as statevec, meas_op
from graphix.sim.statevec import StatevectorBackend as g_svbackend, meas_op as g_meas_op
from graphix import Pattern
from graphix import Circuit
import numpy as np
from src.mbqc import MBQC


def get_pattern(circuit):
    return circuit.standardize_and_transpile()


def get_random_different_from(random_list: list[int], max: int):
    b = np.random.randint(max)

    while b in random_list:
        print(f"get_random_different_from: {random_list}, {b}, {max}")
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
        circuit.i,
    ]
    two_qubit_gates = [circuit.cnot, circuit.swap, circuit.rzz]
    while depth != 0:
        gate_type = 0  # If there is one qubit, we can only use one_qubit gates
        if n == 1:
            pass
        elif n == 2:  # Adapt if there is two qubits
            gate_type = np.random.randint(2)
        else:  # Otherwise we can use all type of gates
            gate_type = np.random.randint(3)

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
        else:  # Three qubits gate
            control1 = np.random.randint(n)
            control2 = get_random_different_from([control1], n)
            target = get_random_different_from([control1, control2], n)
            circuit.ccx(control1, control2, target)
        depth -= 1
    return circuit


def bench_sv_simu(circuit: Circuit):
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
        label1="simu", label2="graphix-simu",
    )
    return bench_simu


def bench_sv_tensor(circuit: Circuit, index: int):
    """
    Compare statevec.prepare_state method with graphix add_nodes method, which
    basically adds new qubit to the statevector in the |+> state.
    """
    p = get_pattern(circuit)

    sv = statevec(len(p.input_nodes))
    to_tensor = np.zeros((2,) * nQubits)
    to_tensor[(0,) * nQubits] = 1

    g_sv = g_svbackend(p)

    bench_tensor = BenchmarkSimu().bench_class_functions(
        sv,
        g_sv,
        (statevec.prepare_state, g_svbackend.add_nodes),
        args1=[index],
        args2=[[index]],
        labels=("sv_tensor", "g_sv_tensor"),
    )
    return bench_tensor


def bench_sv_measure(
    circuit: Circuit,
    node: int,
    angle: int,
    plane: str = "XY",
    s_domain: list[int] = [],
    t_domain: list[int] = [],
    vop=0,
):
    """
    Compare measure methods of simulator and graphix.
    They are not doing the same so maybe this comparison isn't very relevant.
    """
    p = get_pattern(circuit)

    sv = statevec(len(p.input_nodes))
    sv.prepare_state(1)
    sv.entangle(0, 1)

    g_sv = g_svbackend(p)
    g_sv.add_nodes(list(range(p.Nnode)))
    g_sv.entangle_nodes((0, 1))

    bench_measure = BenchmarkSimu().bench_class_functions(
        sv,
        g_sv,
        (statevec.measure, g_svbackend.measure),
        args1=[node, plane, angle, s_domain, t_domain, vop],
        args2=[["M", node, plane, angle, s_domain, t_domain]],
        labels=("sv_measure", "g_measure"),
    )

    return bench_measure


def bench_meas_op(angle, vop, plane, s_signal, t_signal, choice):
    """
    Compare StateVec.meas_op and graphix meas_op.
    """
    bench_meas_op = BenchmarkSimu().bench_functions(
        (meas_op, g_meas_op),
        label1="meas_op",
        label2="g_meas_op",
        args1=[s_signal, t_signal, angle, plane, vop, choice],
        args2=[angle, vop, plane, choice],
    )
    return bench_meas_op


def bench_apply_correction(
    circuit, type: str, index: int, domain: list[int], measurement_results: list[int]
):
    """
    Compare StateVec.apply_correction and graphix correct_byproduct method.
    """
    p = get_pattern(circuit)

    sv = statevec(len(p.input_nodes))
    sv.prepare_state(1)
    sv.entangle(0, 1)

    g_sv = g_svbackend(p)
    g_sv.add_nodes(p.input_nodes)
    g_sv.entangle_nodes((0, 1))
    g_sv.results = measurement_results

    bench_simu = BenchmarkSimu().bench_class_functions(
        obj1=sv,
        obj2=g_sv,
        functions=(statevec.apply_correction, g_svbackend.correct_byproduct),
        args1=[type, index, domain, measurement_results],
        args2=[[type, index, domain, measurement_results]],
        labels=("apply_correction", "graphix-correct_byproduct"),
    )
    return bench_simu


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

circ = build_random_circuit(1, 2)
b_simu = bench_sv_simu(circ)
print(f"bench simu: {b_simu}")
