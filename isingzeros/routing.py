"""
Routing onto a realistic Euro-Q-Exa-like target: the IQM Aphrodite 54-qubit square-lattice fake
backend from `iqm-client[qiskit]` (native gates r, cz; 90 couplers; IQM-supplied error profile).
Euro-Q-Exa is an IQM 54-qubit superconducting system of the same family; the exact coupler list,
calibration and the availability of mid-circuit measurement / reset (`measure` with feedback,
`cc_prx`, `reset` in the IQM instruction set; `reset` "is only available when cc_prx is") must be
confirmed with LRZ at onboarding.

    python -m isingzeros.routing --out results/routing.json

For each representative circuit: two-qubit gate count and depth BEFORE routing (abstract, {rz,sx,x,cz}),
AFTER routing on the Aphrodite coupling map (SWAP overhead), and — for the small cases — the noisy
return probability with the backend's own error profile vs. the exact value.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from qiskit import transpile
from .circuits import kicked_ising_unitary, kicked_ising_lcu, kicked_ising_hadamard_test
from .simulate import exact_amplitude
from .circuits import return_probability_from_counts, lcu_postselect, lcu_normalisation


def target():
    from iqm.qiskit_iqm import IQMFakeAphrodite
    return IQMFakeAphrodite()


def route(qc, backend, basis=('r', 'cz', 'measure', 'reset'), seed=1, trials=3):
    """Transpile with the backend's coupling map; keep reset (not in the fake backend's op list but in
    the IQM instruction set).  Best of `trials` seeds by CZ count."""
    best = None
    for s in range(trials):
        t = transpile(qc, coupling_map=backend.coupling_map, basis_gates=list(basis),
                      optimization_level=3, seed_transpiler=seed + s)
        n2 = dict(t.count_ops()).get('cz', 0)
        if best is None or n2 < best[0]:
            best = (n2, t)
    return best[1]


def abstract_counts(qc):
    t = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cz', 'measure', 'reset'], optimization_level=3, seed_transpiler=1)
    return dict(t.count_ops()).get('cz', 0), t.depth()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/routing.json')
    ap.add_argument('--shots', type=int, default=20000)
    a = ap.parse_args()
    bk = target()
    rows = []
    cases = [
        ('A chain L=8 M=3', kicked_ising_unitary(8, 3, 0.6, 0.6), dict(L=8, M=3, W=1)),
        ('A chain L=12 M=4', kicked_ising_unitary(12, 4, 0.6, 0.6), dict(L=12, M=4, W=1)),
        ('A chain L=16 M=4', kicked_ising_unitary(16, 4, 0.6, 0.6), dict(L=16, M=4, W=1)),
        ('A chain L=24 M=4', kicked_ising_unitary(24, 4, 0.6, 0.6), dict(L=24, M=4, W=1)),
        ('A slab 2x8 M=3', kicked_ising_unitary(8, 3, 0.6, 0.6, W=2, theta3=0.3), dict(L=8, M=3, W=2)),
        ('A slab 3x8 M=3', kicked_ising_unitary(8, 3, 0.6, 0.6, W=3, theta3=0.3), dict(L=8, M=3, W=3)),
        ('A slab 2x12 M=4', kicked_ising_unitary(12, 4, 0.6, 0.6, W=2, theta3=0.3), dict(L=12, M=4, W=2)),
        ('B LCU L=6 M=3 a=0.1 (MCM)', kicked_ising_lcu(6, 3, 0.8, 0.6, 0.1, 0.0), dict(L=6, M=3, W=1, a=0.1)),
        ('B LCU L=8 M=3 a=0.1 (MCM)', kicked_ising_lcu(8, 3, 0.8, 0.6, 0.1, 0.0), dict(L=8, M=3, W=1, a=0.1)),
        ('B LCU L=4 M=2 a=0.1 (deferred)', kicked_ising_lcu(4, 2, 0.8, 0.6, 0.1, 0.0, deferred=True), dict(L=4, M=2, W=1, a=0.1)),
        ('B LCU L=6 M=3 a=0.02 (MCM, no reset)', kicked_ising_lcu(6, 3, 0.63, 0.6, 0.02, 0.0, reset=False), dict(L=6, M=3, W=1, a=0.02)),
        ('B LCU L=6 M=3 a=0.02 (deferred, 24 qubits; sign test without MCM)', kicked_ising_lcu(6, 3, 0.63, 0.6, 0.02, 0.0, deferred=True), dict(L=6, M=3, W=1, a=0.02)),
        ('A chain L=6 M=3 (Fejer-Riesz scan, WP3a)', kicked_ising_unitary(6, 3, 0.6, 0.6), dict(L=6, M=3, W=1)),
        ('C Hadamard L=4 M=2', kicked_ising_hadamard_test(4, 2, 0.6, 0.6), dict(L=4, M=2, W=1)),
    ]
    for name, qc, meta in cases:
        cz0, d0 = abstract_counts(qc)
        t = route(qc, bk)
        ops = dict(t.count_ops())
        row = dict(case=name, **meta, qubits=qc.num_qubits, cz_abstract=cz0, depth_abstract=d0,
                   cz_routed=ops.get('cz', 0), depth_routed=t.depth(), swap_overhead_cz=ops.get('cz', 0) - cz0,
                   ops_routed={k: int(v) for k, v in ops.items()})
        rows.append(row)
        print(f'{name:32s} q={qc.num_qubits:3d}  CZ abstract={cz0:4d} routed={ops.get("cz",0):4d}  depth {d0:4d}->{t.depth():4d}')
    # noisy run with the backend's own error profile for the two smallest protocol-A cases
    for (L, M, th, ph) in [(8, 3, 0.6, 0.6), (8, 4, 0.6, 0.6)]:
        qc = kicked_ising_unitary(L, M, th, ph)
        t = transpile(qc, backend=bk, optimization_level=3, seed_transpiler=1)
        res = bk.run(t, shots=a.shots).result().get_counts()
        p = return_probability_from_counts(res, L)
        ex = abs(exact_amplitude(L, M, th, ph)) ** 2
        rows.append(dict(case=f'noisy A chain L={L} M={M} (Aphrodite error profile)', L=L, M=M, W=1, qubits=L,
                         cz_routed=dict(t.count_ops()).get('cz', 0), depth_routed=t.depth(),
                         p_noisy=p, p_exact=ex, ratio=p / ex if ex else None))
        print(f'noisy L={L} M={M}: P_noisy={p:.4f} P_exact={ex:.4f} ratio={p/ex:.2f}  CZ={dict(t.count_ops()).get("cz",0)}')
    # ring check: every logical bond of the periodic chain, including the closing
    # bond (L-1, 0), must sit on a physical coupler and no SWAP may be inserted
    edges = set(tuple(sorted(e)) for e in bk.coupling_map.get_edges())
    rings = []
    for L in (6, 8, 12, 16, 24):
        qc = kicked_ising_unitary(L, 3, 0.6, 0.6)
        t = route(qc, bk)
        v2p = {qc.qubits.index(v): p for v, p in t.layout.initial_layout.get_virtual_bits().items() if v in qc.qubits}
        mapping = [dict(logical=[i, (i + 1) % L], physical=sorted([v2p[i], v2p[(i + 1) % L]]),
                        is_coupler=tuple(sorted([v2p[i], v2p[(i + 1) % L]])) in edges) for i in range(L)]
        ok = all(m['is_coupler'] for m in mapping) and dict(t.count_ops()).get('cz', 0) == 2 * L * 3
        rings.append(dict(L=L, M=3, cz_routed=dict(t.count_ops()).get('cz', 0), cz_expected_ring=6 * L, all_bonds_on_couplers=ok, mapping=mapping))
        print(f'ring L={L}: all {L} bonds incl. closing bond on physical couplers, no SWAP: {ok}')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'backend': bk.name, 'note': 'IQM Aphrodite 54-qubit FAKE backend from iqm-client: a hardware PROXY, not production Euro-Q-Exa calibration or topology', 'rings': rings, 'num_qubits': bk.num_qubits, 'couplers': len(bk.coupling_map.get_edges()),
               'native_ops': list(bk.operation_names), 'rows': rows}, open(a.out, 'w'), indent=1)
    print('->', a.out)


if __name__ == '__main__':
    main()
