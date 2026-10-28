"""
Emulator benchmark of the planned Euro-Q-Exa circuits (feasibility proof).

    python -m isingzeros.emulate [--traj 200] [--traj-large 100] [--p2 0.005 0.01] [--out results/emulator.json]

For one representative circuit of every campaign: route it on the IQM 54-qubit lattice stand-in (routing.route, as in
Section 4.2 of the ISINGZEROS proposal), keep only the qubits it uses, drop the final measurements and evaluate the probability
that all of them read 0
  - noiseless (statevector), checked against the transfer-matrix value: this validates routing and qubit layout;
  - with gate noise: depolarising p2 after every CZ and p2/10 after every physical single-qubit gate (r, sx, x; rz is
    virtual), averaged over stochastic trajectories (Aer statevector), and exactly with the density matrix up to
    12 qubits as a cross-check.  Readout error is left out: it is corrected separately with confusion matrices.
The angle of each circuit is the largest theta on a coarse grid with P >= 0.05, so that the signal stands well above the
floor 2^-n of a fully depolarised circuit.  The surviving signal fraction F = (P_noisy - 2^-n) / (P_ideal - 2^-n) is
compared with F = (1 - p2)^CZ, the attenuation assumed by the shot budget (budget.resolvable_fraction).
The node wall time per trajectory, times the number of planned points per campaign (results/budget.csv), gives the
CPU node hours that an emulator benchmark of the whole base plan would need.  The JSON is rewritten after every case.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveAmplitudesSquared
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .circuits import kicked_ising_unitary, kicked_ising_lcu, lcu_normalisation
from .simulate import exact_amplitude
from .routing import target, route

PH, TH3, P_MIN = 0.6, 0.3, 0.05
THETAS = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
# (name, campaign, L, M, W, theta3, LCU a) -- one representative circuit per planned circuit family
CASES = [('Ring 4 x 2', 'C_FR', 4, 2, 1, 0.0, None), ('Ring 6 x 3', 'C_FR', 6, 3, 1, 0.0, None),
         ('Ring 4 x 4', 'A2_aspect', 4, 4, 1, 0.0, None), ('Ring 6 x 6', 'A2_aspect', 6, 6, 1, 0.0, None),
         ('Ring 8 x 8', 'A2_aspect', 8, 8, 1, 0.0, None), ('Ring 8 x 3', 'A_chain', 8, 3, 1, 0.0, None),
         ('Ring 8 x 4', 'A_chain', 8, 4, 1, 0.0, None), ('Ring 12 x 4', 'A_chain', 12, 4, 1, 0.0, None),
         ('Ring 16 x 4', 'A_chain', 16, 4, 1, 0.0, None), ('Ring 16 x 6', 'A_chain', 16, 6, 1, 0.0, None),
         ('Slab 2 x 8, 3 rows', 'B_slab', 8, 3, 2, TH3, None), ('Ring 24 x 4', 'A_chain', 24, 4, 1, 0.0, None),
         ('Post-selected 6 x 3, deferred', 'C_sign', 6, 3, 1, 0.0, 0.02),
         ('Slab 2 x 12, 4 rows', 'B_slab', 12, 4, 2, TH3, None), ('Slab 3 x 8, 3 rows', 'B_slab', 8, 3, 3, TH3, None)]


def noise(p2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ['r', 'sx', 'x'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cz'])
    return nm


def compact(t):
    """The routed circuit on the qubits it uses only, without its final measurements (none occur mid-circuit)."""
    active = sorted({t.find_bit(q).index for inst in t.data if inst.operation.name != 'barrier' for q in inst.qubits})
    idx = {a: i for i, a in enumerate(active)}
    c = QuantumCircuit(len(active))
    for inst in t.data:
        if inst.operation.name in ('measure', 'barrier'):
            continue
        assert inst.operation.name not in ('reset',), 'mid-circuit operation in an emulated circuit'
        c.append(inst.operation, [idx[t.find_bit(q).index] for q in inst.qubits])
    c.append(SaveAmplitudesSquared(c.num_qubits, [0], pershot=True), c.qubits)
    return c, [a for a in active]


def p_zero(c, method='statevector', nm=None, shots=1, seed=1, threads=0):
    sim = AerSimulator(method=method, noise_model=nm, seed_simulator=seed, max_parallel_threads=threads)
    t0 = time.perf_counter()
    v = np.asarray(sim.run(c, shots=shots).result().data(0)['amplitudes_squared'], float).reshape(-1)
    return v, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/emulator.json'); ap.add_argument('--budget', default='results/budget.csv')
    ap.add_argument('--traj', type=int, default=200); ap.add_argument('--traj-large', type=int, default=100)
    ap.add_argument('--p2', type=float, nargs='+', default=[0.005, 0.01]); ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    bk = target()
    out = dict(params=vars(a), backend=bk.name, node=os.uname().nodename,
               cpus=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())), cases=[])
    for name, camp, L, M, W, th3, alz in CASES:
        norm = lcu_normalisation(L, M, alz, 0.0, W) ** 2 if alz is not None else 1.0
        for th in THETAS:
            p_exact = abs(exact_amplitude(L, M, th, PH, W=W, theta3=th3, a_zz=alz or 0.0)) ** 2 * norm
            if p_exact >= P_MIN:
                break
        qc = (kicked_ising_lcu(L, M, th, PH, alz, 0.0, W, deferred=True) if alz is not None
              else kicked_ising_unitary(L, M, th, PH, W=W, theta3=th3))
        t0 = time.perf_counter(); t = route(qc, bk); t_route = time.perf_counter() - t0
        c, active = compact(t)
        n = c.num_qubits; ops = dict(c.count_ops())
        p_ideal, t_ideal = p_zero(c, threads=a.threads)
        row = dict(case=name, campaign=camp, L=L, M=M, W=W, theta=th, phi=PH, theta3=th3, a=alz, qubits=n,
                   cz=ops.get('cz', 0), one_qubit=sum(v for k, v in ops.items() if k in ('r', 'sx', 'x')),
                   depth=c.depth(), route_s=t_route, p_exact=p_exact, p_ideal=float(p_ideal[0]),
                   routing_error=abs(float(p_ideal[0]) - p_exact), floor=2.0 ** -n, sv_gb=16 * 2 ** n / 1e9, noisy=[])
        ntraj = a.traj if n <= 12 else a.traj_large
        for p2 in a.p2:
            v, dt = p_zero(c, nm=noise(p2), shots=ntraj, seed=a.seed, threads=a.threads)
            fl = 2.0 ** -n; sig = p_ideal[0] - fl
            r = dict(p2=p2, traj=ntraj, p_mean=float(v.mean()), p_sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                     F=float((v.mean() - fl) / sig), F_sem=float(v.std(ddof=1) / np.sqrt(len(v)) / sig),
                     F_budget=(1 - p2) ** ops.get('cz', 0), wall_s=dt, s_per_traj=dt / ntraj)
            if n <= 12:
                vd, dtd = p_zero(c, method='density_matrix', nm=noise(p2), threads=a.threads)
                r.update(p_dm=float(vd[0]), F_dm=float((vd[0] - fl) / sig), dm_wall_s=dtd)
            row['noisy'].append(r)
        out['cases'].append(row)
        print(f"{name:32s} n={n:2d} CZ={row['cz']:3d} depth={row['depth']:3d} theta={th:.1f} P_exact={p_exact:.4f}  "
              f"|routed-exact|={row['routing_error']:.1e}  " +
              '  '.join(f"p2={r['p2']:.3f}: F={r['F']:.3f}+-{r['F_sem']:.3f}"
                        + (f" (DM {r['F_dm']:.3f})" if 'F_dm' in r else '') + f" budget {r['F_budget']:.3f}"
                        f" {r['s_per_traj']:.3f} s/traj" for r in row['noisy']), flush=True)
        json.dump(out, open(a.out, 'w'), indent=1)
    # node hours to emulate every base point once at each noise level (trajectories above 12 qubits, DM below)
    b = pd.read_csv(a.budget); b = b[(b.system == 'Euro-Q-Exa') & (b.campaign != 'C_zero2D_conditional') & (b.qubits <= 24)]
    by = {(r['L'], r['M'], r['W'], r['a'] is not None): r for r in out['cases']}
    est = []
    for _, g in b.iterrows():
        lcu = g.campaign in ('C_sign', 'C_zero2D')
        r = by[(g.L, g.M, g.W, lcu)]
        if lcu:   # run with mid-circuit measurement on 7 qubits: density matrix; the deferred 24-qubit timing is an upper bound
            per = sum(x['traj'] * x['s_per_traj'] for x in r['noisy'])
        elif r['qubits'] <= 12:
            per = sum(x['dm_wall_s'] for x in r['noisy'])
        else:
            per = sum(x['traj'] * x['s_per_traj'] for x in r['noisy'])
        est.append(dict(campaign=g.campaign, L=int(g.L), M=int(g.M), W=int(g.W), qubits=int(g.qubits),
                        points=int(g.points), node_h=int(g.points) * per / 3600))
    out['estimate'] = dict(rows=est, node_hours=sum(e['node_h'] for e in est),
                           note='one noisy emulation of every base point at each p2 level, measured node time')
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f"base plan, every point at {len(a.p2)} noise levels: {out['estimate']['node_hours']:.1f} node hours "
          f"({out['cpus']} cores per node)")
    print('->', a.out)


if __name__ == '__main__':
    main()
