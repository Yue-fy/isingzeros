"""
Quantum circuits (Qiskit) that measure the strip partition function of the 2D / layered Ising model
at complex couplings.

Geometry: a row is a W x L sheet of qubits (W = 1: chain, i.e. the 2D classical model;
W = 2, 3: ladder, i.e. a layered 3D classical model with W layers).  Qubit index q = w*L + k.

Operator realised:   U = V_y (V_x V_y)^{M-1},
    V_y = exp( K2 sum_k Z_k Z_{k+1} + K3 sum_rungs Z Z ),   V_x = exp( K1* sum_q X_q ).

(a) Purely imaginary couplings  K2 = i*theta, K3 = i*theta3, K1* = i*phi  ->  U is unitary:
        exp(i theta Z Z) = RZZ(-2 theta),   exp(i phi X) = RX(-2 phi).
    Observable: the return probability  P = |<+^n| U |+^n>|^2  (prepare |+>, apply U, measure in the
    X basis, count the all-"+" outcome).  Relation to the strip partition function:
        Z_strip = (2 sinh 2K1)^{n(M-1)/2} 2^n <+|U|+>,   so   |Z_strip| = |2 sinh 2K1|^{n(M-1)/2} 2^n sqrt(P).
    Zeros of Z_strip on the imaginary-coupling family are the zeros of P.

(b) Complex phase: Hadamard test with one ancilla gives <+|U|+> itself (real part with ancilla |+>,
    imaginary part with ancilla |+i>).  Needs controlled-RZZ / controlled-RX, i.e. ~3x the two-qubit
    gate count; reserved for small sizes.

(c) Real parts of the couplings (leaving the imaginary axis).  exp(a ZZ) = cosh a + sinh a ZZ is a
    linear combination of two unitaries.  With one ancilla prepared in (sqrt(cosh a)|0> + sqrt(sinh a)|1>)
    / sqrt(e^a), a controlled-ZZ, the inverse preparation and post-selection of the ancilla on |0>, the
    register is acted on by e^{-a} (cosh a + sinh a ZZ) = e^{-a} exp(a ZZ).  The ancilla is reset and
    reused bond by bond (mid-circuit measurement); the overall success probability is
    prod_bonds e^{-2 a}|...|^2 ~ exp(-2 sum |a|), cheap for small real parts.  Same for exp(b X).
    This is how the circuit reaches a strip around the imaginary axis in the complex coupling plane.

All circuits return (QuantumCircuit, metadata).  Transpilation to the target's native gates is done by
the caller (see run_aer.py / run_hardware.py).
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RZZGate, RXGate, RYGate, RZGate


def bonds(L: int, W: int, periodic_legs: bool = False):
    """List of (q, q', kind): kind 'leg' for bonds along a leg (periodic in L), 'rung' across legs."""
    out = []
    for w in range(W):
        for k in range(L):
            out.append((w * L + k, w * L + (k + 1) % L, 'leg'))
    for w in range(W - 1 if not periodic_legs else W):
        for k in range(L):
            out.append((w * L + k, ((w + 1) % W) * L + k, 'rung'))
    return out


# ----------------------------------------------------------------------------- (a) unitary circuit
def kicked_ising_unitary(L: int, M: int, theta: float, phi: float, W: int = 1, theta3: float = 0.0,
                         periodic_legs: bool = False, measure: bool = True) -> QuantumCircuit:
    """|+^n> -> V_y (V_x V_y)^{M-1} |+^n>, then X-basis measurement.  Imaginary couplings only."""
    n = L * W
    qr = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qr, ClassicalRegister(n, 'c')) if measure else QuantumCircuit(qr)
    qc.h(qr)
    bl = bonds(L, W, periodic_legs)

    def Vy():
        # leg bonds: even then odd (commute anyway, ordering only matters for depth)
        for par in (0, 1):
            for (a, b, kind) in bl:
                if kind == 'leg' and (a % L) % 2 == par:
                    qc.append(RZZGate(-2.0 * theta), [qr[a], qr[b]])
        if W > 1 and theta3 != 0.0:
            for (a, b, kind) in bl:
                if kind == 'rung':
                    qc.append(RZZGate(-2.0 * theta3), [qr[a], qr[b]])

    def Vx():
        for q in range(n):
            qc.append(RXGate(-2.0 * phi), [qr[q]])

    Vy()
    for _ in range(M - 1):
        Vx(); Vy()
    if measure:
        qc.h(qr)
        qc.measure(qr, qc.clbits)
    return qc


def return_probability_from_counts(counts: dict, n: int) -> float:
    tot = sum(counts.values())
    return counts.get('0' * n, 0) / tot if tot else float('nan')


# ----------------------------------------------------------------------------- (b) Hadamard test
def kicked_ising_hadamard_test(L: int, M: int, theta: float, phi: float, W: int = 1, theta3: float = 0.0,
                               imag_part: bool = False) -> QuantumCircuit:
    """Ancilla-controlled U; <X_anc> = Re <+|U|+>, <Y_anc> (imag_part=True) = Im <+|U|+>."""
    n = L * W
    qr = QuantumRegister(n, 'q'); anc = QuantumRegister(1, 'a')
    qc = QuantumCircuit(anc, qr, ClassicalRegister(1, 'c'))
    qc.h(qr); qc.h(anc[0])
    bl = bonds(L, W)

    def cVy():
        for (a, b, kind) in bl:
            if kind == 'leg':
                qc.append(RZZGate(-2.0 * theta).control(1), [anc[0], qr[a], qr[b]])
            elif theta3 != 0.0:
                qc.append(RZZGate(-2.0 * theta3).control(1), [anc[0], qr[a], qr[b]])

    def cVx():
        for q in range(n):
            qc.append(RXGate(-2.0 * phi).control(1), [anc[0], qr[q]])

    cVy()
    for _ in range(M - 1):
        cVx(); cVy()
    if imag_part:
        qc.sdg(anc[0])
    qc.h(anc[0])
    qc.measure(anc[0], 0)
    return qc


def hadamard_expectation_from_counts(counts: dict) -> float:
    tot = sum(counts.values())
    return (counts.get('0', 0) - counts.get('1', 0)) / tot if tot else float('nan')


# ----------------------------------------------------------------------------- (c) real parts via LCU
def _lcu_prep_angle(a: float) -> float:
    """Rotation angle ry(alpha) with cos(alpha/2) = sqrt(cosh a / e^a), sin(alpha/2) = sqrt(sinh a / e^a)."""
    a = abs(a)
    c = np.sqrt(np.cosh(a) / np.exp(a))
    return 2.0 * np.arccos(np.clip(c, -1, 1))


def kicked_ising_lcu(L: int, M: int, theta: float, phi: float, a_zz: float, b_x: float,
                     W: int = 1, theta3: float = 0.0, deferred: bool = False, reset: bool = True,
                     zz_steps_at_zero: bool = True) -> QuantumCircuit:
    """Complex couplings K2 = a_zz + i theta, K1* = b_x + i phi (real parts via post-selected LCU).

    For a real parameter a of either sign write  exp(a ZZ) = cosh|a| I + sgn(a) sinh|a| ZZ.
    One ancilla is prepared in  (sqrt(cosh|a|)|0> + sqrt(sinh|a|)|1>) / sqrt(e^{|a|})  by RY(alpha),
    cos(alpha/2) = sqrt(cosh|a| / e^{|a|}); for a < 0 a Z gate on the ancilla multiplies the |1> branch
    by -1 (controlled-(-ZZ)); then controlled-ZZ (two CZ), RY(-alpha), measurement.  Outcome 0 leaves
    the register acted on by  e^{-|a|} (cosh|a| + sgn(a) sinh|a| ZZ) = e^{-|a|} exp(a ZZ).  The same
    construction with controlled-X gives exp(b X).  After post-selecting ALL ancilla outcomes on 0
    the register is in
        N * V_y (V_x V_y)^{M-1} |+>,   N = exp(-(#ZZ steps) |a_zz| - (#X steps) |b_x|),
    and the final X-basis measurement gives the return probability.  Estimator (see simulate.py):
        |<+|U(K)|+>|^2 = P(return | all ancillas 0) * P(all ancillas 0) / N^2 .

    deferred=False: one ancilla, mid-circuit measurement + reset after every step (needs MCM/reset
                    support on the backend).
    deferred=True : one FRESH ancilla per step, all measured at the end (no mid-circuit measurement;
                    qubit count n + #steps, usable only for small L, M; functionally identical).
    The classical register 'anc' holds one bit per LCU step in both variants.
    reset=False   : MCM variant WITHOUT reset.  In the post-selected ensemble every kept shot has read 0
                    on the ancilla, which then is already |0> (ideal projective measurement), so the reset
                    is redundant for the estimator; only mid-circuit measurement + per-shot records are
                    needed.  (Differences appear only through readout errors; see verify.py V7.)
    zz_steps_at_zero=True : keep the ZZ ancilla steps also at a_zz = 0 (alpha = 0).  A grid in a that crosses
                    a = 0 then uses one circuit structure, so its noise does not jump on the a = 0 column.
    """
    n = L * W
    bl = bonds(L, W)
    zz_on = a_zz != 0.0 or zz_steps_at_zero
    n_zz_steps = (M * len([b for b in bl if b[2] == 'leg'])) if zz_on else 0
    n_x_steps = ((M - 1) * n) if b_x != 0.0 else 0
    n_steps = n_zz_steps + n_x_steps
    qr = QuantumRegister(n, 'q'); anc = QuantumRegister(max(1, n_steps) if deferred else 1, 'a')
    cr = ClassicalRegister(n, 'c'); ca = ClassicalRegister(max(1, n_steps), 'anc')
    qc = QuantumCircuit(anc, qr, cr, ca)
    qc.h(qr)
    step = [0]
    alpha_zz = _lcu_prep_angle(a_zz); alpha_x = _lcu_prep_angle(b_x)

    def A():   # ancilla for the current step
        return anc[step[0]] if deferred else anc[0]

    def finish():
        if deferred:
            qc.measure(A(), ca[step[0]])
        else:
            qc.measure(A(), ca[step[0]])
            if reset:
                qc.reset(A())
        step[0] += 1

    def lcu_zz(a, b):
        qc.append(RZZGate(-2.0 * theta), [qr[a], qr[b]])        # unitary part exp(i theta ZZ)
        if not zz_on:
            return
        qc.ry(alpha_zz, A())
        if a_zz < 0:
            qc.z(A())                                            # controlled-(-ZZ)
        qc.cz(A(), qr[a]); qc.cz(A(), qr[b])                     # controlled-ZZ
        qc.ry(-alpha_zz, A())
        finish()

    def lcu_x(q):
        qc.append(RXGate(-2.0 * phi), [qr[q]])
        if b_x == 0.0:
            return
        qc.ry(alpha_x, A())
        if b_x < 0:
            qc.z(A())
        qc.cx(A(), qr[q])                                        # controlled-X
        qc.ry(-alpha_x, A())
        finish()

    def Vy():
        for (a, b, kind) in bl:
            if kind == 'leg':
                lcu_zz(a, b)
            elif theta3 != 0.0:
                qc.append(RZZGate(-2.0 * theta3), [qr[a], qr[b]])

    def Vx():
        for q in range(n):
            lcu_x(q)

    Vy()
    for _ in range(M - 1):
        Vx(); Vy()
    qc.h(qr)
    qc.measure(qr, cr)
    return qc


def lcu_postselect(counts: dict, n: int) -> tuple[float, float]:
    """From counts with keys 'anc_bits c_bits' (Qiskit prints registers space-separated, last-declared
    first): returns (success_probability, return_probability_given_success)."""
    acc = 0; ok = 0; tot = 0
    for key, c in counts.items():
        parts = key.split()
        anc_bits, c_bits = parts[0], parts[1]
        tot += c
        if set(anc_bits) <= {'0'}:
            acc += c
            if c_bits == '0' * n:
                ok += c
    return (acc / tot if tot else float('nan'), ok / acc if acc else float('nan'))


def lcu_normalisation(L: int, M: int, a_zz: float, b_x: float, W: int = 1) -> float:
    """N = exp(-(#ZZ steps)|a| - (#X steps)|b|): the register amplitude is N * exp(a ZZ)... applied."""
    n = L * W
    n_zz = M * (L * W)        # leg bonds per V_y times M applications of V_y
    n_x = (M - 1) * n
    return np.exp(-n_zz * abs(a_zz) - n_x * abs(b_x))


_ANGLE_GATES = {'rzz': RZZGate, 'rx': RXGate, 'ry': RYGate, 'rz': RZGate}


def compile_fixed(qc: QuantumCircuit, basis=('cz', 'rz', 'sx', 'x', 'h', 'reset', 'measure'), optimization_level=1):
    """Transpile with every rotation angle replaced by a symbolic parameter, then bind the numerical values.
    The compiled gate sequence is then the same for all angles: no angle-specific simplification such as
    RZZ(0) -> identity or merged Clifford rotations.  This is what a parametric template compiled once does
    on hardware, and it keeps a gate-level noise model from depending on the angles."""
    from qiskit import transpile
    from qiskit.circuit import Parameter
    tmpl = qc.copy_empty_like()
    values = {}
    for i, inst in enumerate(qc.data):
        op = inst.operation
        if op.name in _ANGLE_GATES and not op.is_parameterized():
            p = Parameter(f'_a{i}')
            values[p] = float(op.params[0])
            tmpl.append(_ANGLE_GATES[op.name](p), inst.qubits, inst.clbits)
        else:
            tmpl.append(op, inst.qubits, inst.clbits)
    t = transpile(tmpl, basis_gates=list(basis), optimization_level=optimization_level)
    return t.assign_parameters({p: v for p, v in values.items() if p in t.parameters})


def gate_counts(qc: QuantumCircuit, basis=('cz', 'rz', 'sx', 'x', 'h', 'measure', 'reset')) -> dict:
    from qiskit import transpile
    t = transpile(qc, basis_gates=list(basis), optimization_level=1)
    ops = dict(t.count_ops())
    return {'two_qubit': ops.get('cz', 0), 'depth': t.depth(), 'ops': ops}
