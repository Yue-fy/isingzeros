"""Exact (statevector) and noisy (Aer) evaluation of the circuits of circuits.py, and the
cross-check against the transfer matrix of classical.py."""
from __future__ import annotations
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from .classical import RowTransfer, circuit_couplings
from .circuits import (kicked_ising_unitary, kicked_ising_hadamard_test, kicked_ising_lcu,
                       return_probability_from_counts, hadamard_expectation_from_counts,
                       lcu_postselect, lcu_normalisation, compile_fixed)


# ----------------------------------------------------------------------------- exact references
def exact_amplitude(L, M, theta, phi, W=1, theta3=0.0, a_zz=0.0, b_x=0.0):
    """<+|V_y (V_x V_y)^{M-1}|+> from the transfer matrix at K2 = a_zz + i theta, K1* = b_x + i phi."""
    from .classical import primal_coupling
    K2 = a_zz + 1j * theta
    K1 = primal_coupling(b_x + 1j * phi)
    K3 = 1j * theta3
    T = RowTransfer(L, K1, K2, W=W, K3=K3)
    return T.strip_amplitude(M)


def statevector_amplitude(L, M, theta, phi, W=1, theta3=0.0):
    qc = kicked_ising_unitary(L, M, theta, phi, W=W, theta3=theta3, measure=False)
    n = L * W
    psi = Statevector(qc)
    plus = Statevector.from_label('+' * n)
    return complex(plus.inner(psi))


# ----------------------------------------------------------------------------- noise models
def generic_noise(p2: float = 5e-3, p1: float = 5e-4, p_ro: float = 1e-2) -> NoiseModel:
    """Depolarizing two-qubit error p2 (per CZ/RZZ), one-qubit error p1, symmetric readout error p_ro.
    p2 = 5e-3 ~ a good 2026 superconducting device; p2 = 2e-3 ~ trapped ions."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rx', 'ry', 'sx', 'x', 'h'])     # rz: virtual, error-free
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cz', 'cx', 'rzz'])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def run_counts(qc, shots=10000, noise: NoiseModel | None = None, seed=1, basis=('cz', 'rz', 'sx', 'x', 'h', 'reset', 'measure')):
    sim = AerSimulator(noise_model=noise, seed_simulator=seed) if noise is not None else AerSimulator(seed_simulator=seed)
    t = compile_fixed(qc, basis)     # angle-independent gate sequence (circuits.compile_fixed)
    return sim.run(t, shots=shots).result().get_counts(), t


# ----------------------------------------------------------------------------- end-to-end evaluations
def return_probability(L, M, theta, phi, W=1, theta3=0.0, shots=10000, noise=None, seed=1):
    qc = kicked_ising_unitary(L, M, theta, phi, W=W, theta3=theta3)
    counts, t = run_counts(qc, shots, noise, seed)
    return return_probability_from_counts(counts, L * W), t


def hadamard_amplitude(L, M, theta, phi, W=1, theta3=0.0, shots=10000, noise=None, seed=1):
    re_c, t = run_counts(kicked_ising_hadamard_test(L, M, theta, phi, W, theta3, False), shots, noise, seed)
    im_c, _ = run_counts(kicked_ising_hadamard_test(L, M, theta, phi, W, theta3, True), shots, noise, seed + 1)
    return hadamard_expectation_from_counts(re_c) + 1j * hadamard_expectation_from_counts(im_c), t


def lcu_return_probability(L, M, theta, phi, a_zz, b_x, W=1, theta3=0.0, shots=20000, noise=None, seed=1):
    """Returns (estimate of |<+|U(K)|+>|^2 with complex couplings, success probability, circuit).
    With post-selection the accepted register state is N U|+> / sqrt(P_succ); hence
        P_return|succ = N^2 |<+|U|+>|^2 / P_succ,  i.e.  |<+|U|+>|^2 = P_return|succ * P_succ / N^2."""
    qc = kicked_ising_lcu(L, M, theta, phi, a_zz, b_x, W, theta3)
    counts, t = run_counts(qc, shots, noise, seed)
    p_succ, p_ret = lcu_postselect(counts, L * W)
    N = lcu_normalisation(L, M, a_zz, b_x, W)
    return p_ret * p_succ / N ** 2, p_succ, t
